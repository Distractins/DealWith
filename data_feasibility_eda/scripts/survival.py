# -*- coding: utf-8 -*-
"""④ 生存附属分析: KM 曲线 + log-rank + Cox EPV (仅作事后验证)。"""
import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from loader import COLOR_NONMUC, COLOR_MUC


def run_survival(df, cohort, out_fig, out_tab, plt, cfg):
    d = df.dropna(subset=["time", "event"]).copy()
    d["event"] = d["event"].astype(int)
    n_total = len(d)
    n_event = int(d["event"].sum())
    n_censored = int((d["event"] == 0).sum())
    censoring_rate = n_censored / n_total if n_total else np.nan
    epv = n_event / cfg["survival_n_predictors"]

    muc = d[d["is_mucinous"]]
    non = d[~d["is_mucinous"]]

    logrank_p = np.nan
    if len(muc) > 0 and len(non) > 0 and muc["event"].sum() > 0 and non["event"].sum() > 0:
        res = logrank_test(muc["time"], non["time"], muc["event"], non["event"])
        logrank_p = res.p_value

    _plot_km(d, muc, non, cohort, out_fig, plt)

    # 单变量 Cox (粘液 vs 非粘液), 事件太少则跳过
    hr, hr_p = np.nan, np.nan
    if n_event >= 10:
        try:
            dcox = d[["time", "event", "is_mucinous"]].copy()
            dcox["is_mucinous"] = dcox["is_mucinous"].astype(int)
            cph = CoxPHFitter()
            cph.fit(dcox, duration_col="time", event_col="event")
            hr = float(np.exp(cph.params_["is_mucinous"]))
            hr_p = float(cph.summary.loc["is_mucinous", "p"])
        except Exception:
            pass

    surv = {
        "n_total": n_total, "n_event": n_event, "n_censored": n_censored,
        "censoring_rate": censoring_rate, "epv": epv,
        "logrank_p": logrank_p, "cox_hr": hr, "cox_hr_p": hr_p,
    }
    pd.DataFrame([{"指标": k, "值": v} for k, v in surv.items()]).to_csv(
        os.path.join(out_tab, "04_survival_summary.csv"), index=False, encoding="utf-8-sig")
    return surv


def _plot_km(d, muc, non, cohort, out_fig, plt):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    kmf = KaplanMeierFitter()
    kmf.fit(non["time"], non["event"], label=f"非粘液 (n={len(non)}, 事件{int(non['event'].sum())})")
    kmf.plot_survival_function(ax=ax, color=COLOR_NONMUC)
    if len(muc) > 0:
        kmf.fit(muc["time"], muc["event"], label=f"粘液 (n={len(muc)}, 事件{int(muc['event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=COLOR_MUC)
    ax.set_xlabel("生存时间 (月)")
    ax.set_ylabel("生存概率")
    ax.set_title(f"{cohort} 粘液 vs 非粘液 生存 (KM)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "04_km_curves.png"), bbox_inches="tight")
    plt.close(fig)
