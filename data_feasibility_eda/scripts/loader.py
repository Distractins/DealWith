# -*- coding: utf-8 -*-
"""配置解析 + 数据加载 + 绘图风格。所有路径相对仓库根解析。"""
import os
import yaml
import numpy as np
import pandas as pd

MODULE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))


def resolve(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def load_config():
    with open(os.path.join(MODULE_DIR, "config", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_dir(cfg, cohort, kind):
    d = os.path.join(MODULE_DIR, cfg["output_dir"], cohort.lower(), kind)
    os.makedirs(d, exist_ok=True)
    return d


def load_cohort(cfg, name):
    """读取单个队列, 统一字段: patient_id / is_mucinous / event / time / censored / 基因。"""
    c = cfg["cohorts"][name]
    df = pd.read_csv(resolve(c["csv"]), encoding="utf-8-sig")
    df = df.rename(columns={cfg["patient_id_col"]: "patient_id"})

    # 粘液标注: 含 "mucinous"(不区分大小写) 或命中别名
    aliases = set(c.get("mucinous_aliases", []))

    def _is_muc(x):
        x = str(x).strip()
        return ("mucinous" in x.lower()) or (x in aliases)

    df["is_mucinous"] = df[cfg["histology_col"]].apply(_is_muc)

    # 生存: event(1=死亡) / time(月) / censored(1=删失)
    df["event"] = pd.to_numeric(df.get("event"), errors="coerce")
    df["time"] = pd.to_numeric(df.get("Survival months"), errors="coerce")
    cens_col = c.get("censored_col")
    if cens_col and cens_col in df.columns:
        df["censored"] = pd.to_numeric(df[cens_col], errors="coerce")
    else:
        df["censored"] = 1 - df["event"]

    # 基因: 取当前队列实际存在的目标基因, 数值化, 按需 log1p
    genes = [g for g in cfg["gene_list"] if g in df.columns]
    missing = [g for g in cfg["gene_list"] if g not in df.columns]
    for g in genes:
        df[g] = pd.to_numeric(df[g], errors="coerce")
        if c.get("gene_log1p"):
            df[g] = np.log1p(df[g])
    gene_unit = "log1p(TPM)" if c.get("gene_log1p") else "log2(TPM+1)"
    return df, genes, missing, gene_unit


# ---- 绘图风格 (论文补充材料: 高清 + 简洁) ----
COLOR_MUC = "#E69F00"     # 粘液: 橙
COLOR_NONMUC = "#0072B2"  # 非粘液: 蓝


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    return plt
