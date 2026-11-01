# -*- coding: utf-8 -*-
"""
fetch_gdc_mapping.py - 从 GDC API 导出「文件UUID -> TCGA条形码」映射表, 整理成干净4列。

用法:
    python fetch_gdc_mapping.py TCGA-READ
    python fetch_gdc_mapping.py TCGA-COAD

输出:
    {project}_file_case_map.tsv  (file_id / data_category / data_type / case_submitter_id)
"""

import sys
import json
import csv
import io
import urllib.request
import urllib.parse

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "TCGA-READ"

filters = {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT]}}
fields = "file_id,file_name,data_category,data_type,cases.submitter_id"

url = "https://api.gdc.cancer.gov/files"
params = urllib.parse.urlencode({
    "filters": json.dumps(filters),
    "fields": fields,
    "format": "TSV",
    "size": "10000",
})

print(f"[fetch] 查询 GDC API: {PROJECT} ...")
req = urllib.request.Request(url + "?" + params, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=120) as resp:
    text = resp.read().decode("utf-8")

# 解析 GDC 返回的宽表, 提取 file_id / data_category / data_type / 第一个非空 case submitter_id
rows = []
reader = csv.DictReader(io.StringIO(text), delimiter="\t")
for r in reader:
    # 找第一个非空的 cases.*.submitter_id
    case_id = ""
    for i in range(200):
        key = f"cases.{i}.submitter_id"
        if key in r and r[key].strip():
            case_id = r[key].strip()
            break
    rows.append({
        "file_id": r.get("file_id", "").strip(),
        "data_category": r.get("data_category", "").strip(),
        "data_type": r.get("data_type", "").strip(),
        "case_submitter_id": case_id,
    })

# 去重 (同一file_id只留一条)
seen = set()
clean_rows = []
for r in rows:
    if r["file_id"] and r["file_id"] not in seen:
        seen.add(r["file_id"])
        clean_rows.append(r)

out = f"{PROJECT}_file_case_map.tsv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file_id", "data_category", "data_type", "case_submitter_id"], delimiter="\t")
    w.writeheader()
    w.writerows(clean_rows)

print(f"[fetch] 去重后共 {len(clean_rows)} 个文件映射, 已保存: {out}")
print("[fetch] 前 8 行:")
for r in clean_rows[:8]:
    print(f"  {r['file_id'][:12]}...  {r['data_category'][:28]:28s}  {r['case_submitter_id']}")
