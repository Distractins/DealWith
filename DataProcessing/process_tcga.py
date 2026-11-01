# -*- coding: utf-8 -*-
"""
process_tcga.py - 把 GDC 原始数据(rar) 处理成 mucus_mil_regression 需要的 CSV。

输入:
    DataProcessing/{PROJECT}/TCGA_{PROJECT}_CLINICAL.rar   (临床XML)
    DataProcessing/{PROJECT}/TCGA_{PROJECT}_RNA.rar        (RNA counts TSV)
    DataProcessing/{PROJECT}/TCGA_{PROJECT}_MUT.rar        (突变 MAF.gz)
    {PROJECT}_file_case_map.tsv                            (file_id -> TCGA条形码 映射)

输出:
    DataProcessing/{PROJECT}/{PROJECT}_processed.csv
        TCGA ID / 临床特征 / DNA突变 / 4个目标基因RNA-seq表达

用法:
    python process_tcga.py READ
    python process_tcga.py COAD   (等COAD下好后复用)
"""

import os
import sys
import csv
import gzip
import io
import glob
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "READ"
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, PROJECT)
EXTRACT_DIR = os.path.join(DATA_DIR, "extracted")
MAPPING = os.path.join(BASE, f"TCGA-{PROJECT}_file_case_map.tsv")

# 4 个目标基因 (PCA PC1 用)
TARGET_GENES = ["REG1A", "REG4", "SPINK4", "ITLN1"]
# 关键 DNA 突变基因
KEY_MUT_GENES = ["KRAS", "BRAF", "APC", "TP53", "PIK3CA", "NRAS", "SMAD4", "FBXW7"]
# 非同义突变类型
NONSYN = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
          "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
          "Nonstop_Mutation", "Start_Codon_Del", "Start_Codon_SNP", "Stop_Codon_Del"}


def extract_rar(rar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[extract] 解压 {os.path.basename(rar_path)} ...")
    subprocess.run(["bsdtar", "-xf", rar_path, "-C", out_dir], check=True)


def iter_local(root, tag):
    """遍历XML, 按本地标签名(忽略命名空间)找元素。"""
    for elem in root.iter():
        if elem.tag.split('}')[-1] == tag:
            yield elem


def first_text(root, tag):
    for elem in iter_local(root, tag):
        if elem.text and elem.text.strip():
            return elem.text.strip()
    return ""


# ============================================================
# 1. 解析临床 XML
# ============================================================
def parse_clinical(xml_dir):
    clinical = {}
    xml_files = glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)
    print(f"[clinical] 解析 {len(xml_files)} 个临床XML ...")
    for xf in xml_files:
        try:
            root = ET.parse(xf).getroot()
        except Exception:
            continue
        barcode = first_text(root, "bcr_patient_barcode")
        if not barcode or not barcode.startswith("TCGA-"):
            continue
        age = first_text(root, "age_at_initial_pathologic_diagnosis")
        gender = first_text(root, "gender")
        hist = first_text(root, "histological_type")
        stage = first_text(root, "pathologic_stage")
        pt = first_text(root, "pathologic_T")
        pn = first_text(root, "pathologic_N")
        pm = first_text(root, "pathologic_M")
        vital = first_text(root, "vital_status")
        days_death = first_text(root, "days_to_death")
        days_followup = first_text(root, "days_to_last_followup")

        # 生存事件与月份 (仅辅助, 本任务主指标不用)
        if vital.lower() == "dead" and days_death:
            event = 1
            surv_months = float(days_death) / 30.4
        elif days_followup:
            event = 0
            surv_months = float(days_followup) / 30.4
        else:
            event = 0
            surv_months = ""

        clinical[barcode] = {
            "age_at_diagnosis": age, "gender": gender, "histological_type": hist,
            "tumor_stage": stage, "t": pt, "n": pn, "m": pm,
            "event": event, "Survival months": surv_months,
        }
    print(f"[clinical] 有效病人: {len(clinical)}")
    return clinical


# ============================================================
# 2. 解析 RNA (提取4个目标基因 TPM)
# ============================================================
def parse_rna(rna_dir, mapping):
    rna = defaultdict(dict)  # barcode -> {gene: tpm}
    count = 0
    for file_id, barcode in mapping.items():
        folder = os.path.join(rna_dir, file_id)
        if not os.path.isdir(folder):
            continue
        tsvs = glob.glob(os.path.join(folder, "*.rna_seq*.tsv"))
        if not tsvs:
            continue
        tsv = tsvs[0]
        try:
            gene_tpm = {}
            with open(tsv, encoding="utf-8", errors="ignore") as f:
                # 手动解析 (不用csv模块, 避免 decoder 报错), 跳过 '#' 注释行
                header = None
                for line in f:
                    if line.startswith("#"):
                        continue
                    header = line.rstrip("\r\n").split("\t")
                    break
                if header is None:
                    continue
                try:
                    idx_gene = header.index("gene_name")
                    idx_tpm = header.index("tpm_unstranded")
                except ValueError:
                    continue
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) <= idx_tpm:
                        continue
                    gn = parts[idx_gene]
                    if gn in TARGET_GENES:
                        try:
                            gene_tpm[gn] = float(parts[idx_tpm])
                        except ValueError:
                            gene_tpm[gn] = 0.0
            for g in TARGET_GENES:
                rna[barcode][g] = gene_tpm.get(g, 0.0)
            count += 1
        except Exception as e:
            print(f"  [rna] 跳过 {file_id[:8]}: {e}")
    print(f"[rna] 解析 {count} 个RNA样本")
    return rna


# ============================================================
# 3. 解析 MAF (提取关键基因突变 + TMB)
# ============================================================
def parse_maf(mut_dir, mapping):
    mut = defaultdict(lambda: {"genes": set(), "tmb": 0})  # barcode -> 突变基因集合 + TMB
    count = 0
    for file_id, barcode in mapping.items():
        folder = os.path.join(mut_dir, file_id)
        if not os.path.isdir(folder):
            continue
        mafs = glob.glob(os.path.join(folder, "*.maf.gz")) + glob.glob(os.path.join(folder, "*.maf"))
        if not mafs:
            continue
        maf = mafs[0]
        try:
            opener = gzip.open if maf.endswith(".gz") else open
            with opener(maf, "rt", encoding="utf-8", errors="ignore") as f:
                # 跳过注释行
                lines = (l for l in f if not l.startswith("#"))
                reader = csv.DictReader(lines, delimiter="\t")
                n_nonsyn = 0
                for row in reader:
                    gene = row.get("Hugo_Symbol", "")
                    cls = row.get("Variant_Classification", "")
                    if cls in NONSYN:
                        n_nonsyn += 1
                        if gene in KEY_MUT_GENES:
                            mut[barcode]["genes"].add(gene)
                mut[barcode]["tmb"] += n_nonsyn
            count += 1
        except Exception as e:
            print(f"  [maf] 跳过 {file_id[:8]}: {e}")
    print(f"[maf] 解析 {count} 个突变样本")
    return mut


# ============================================================
# 主流程
# ============================================================
def main():
    # 0) 解压 (若已解压则跳过)
    for key in ["CLINICAL", "RNA", "MUT"]:
        rar = os.path.join(DATA_DIR, f"TCGA_{PROJECT}_{key}.rar")
        target = os.path.join(EXTRACT_DIR, f"TCGA_{PROJECT}_{key}")
        if not os.path.exists(rar):
            continue
        if os.path.isdir(target) and os.listdir(target):
            print(f"[skip] 已解压: {target}")
            continue
        extract_rar(rar, EXTRACT_DIR)

    # 加载映射 (file_id -> TCGA barcode), 只保留 RNA 和 突变
    mapping = {}
    with open(MAPPING, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cat = row.get("data_category", "")
            if cat in ("Transcriptome Profiling", "Simple Nucleotide Variation"):
                mapping[row["file_id"]] = row["case_submitter_id"]
    print(f"[map] RNA+突变 文件映射: {len(mapping)} 个")

    # 1) 临床
    clinical = parse_clinical(os.path.join(EXTRACT_DIR, f"TCGA_{PROJECT}_CLINICAL"))

    # 2) RNA
    rna = parse_rna(os.path.join(EXTRACT_DIR, f"TCGA_{PROJECT}_RNA"), mapping)

    # 3) 突变
    mut = parse_maf(os.path.join(EXTRACT_DIR, f"TCGA_{PROJECT}_MUT"), mapping)

    # 4) join
    all_barcodes = set(clinical) | set(rna) | set(mut)
    columns = ["TCGA ID", "event", "Survival months", "age_at_diagnosis", "gender",
               "tumor_stage", "t", "n", "m", "histological_type"]
    columns += [f"{g}_mut" for g in KEY_MUT_GENES] + ["TMB"]
    columns += [f"{g}_rnaseq" for g in TARGET_GENES]

    rows = []
    for bc in sorted(all_barcodes):
        row = {"TCGA ID": bc}
        c = clinical.get(bc, {})
        for k in ["event", "Survival months", "age_at_diagnosis", "gender",
                  "tumor_stage", "t", "n", "m", "histological_type"]:
            row[k] = c.get(k, "")
        for g in KEY_MUT_GENES:
            row[f"{g}_mut"] = 1 if (g in mut.get(bc, {}).get("genes", set())) else 0
        row["TMB"] = mut.get(bc, {}).get("tmb", 0)
        for g in TARGET_GENES:
            row[f"{g}_rnaseq"] = rna.get(bc, {}).get(g, "")
        rows.append(row)

    # 5) 输出
    out_csv = os.path.join(DATA_DIR, f"{PROJECT}_processed.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[完成] 输出: {out_csv}  ({len(rows)} 个病人)")
    print(f"[完成] 有RNA表达的病人: {len(rna)}, 有突变的病人: {len(mut)}, 有临床的病人: {len(clinical)}")


if __name__ == "__main__":
    main()
