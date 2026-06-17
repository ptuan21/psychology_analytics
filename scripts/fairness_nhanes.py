"""
#2 CROSS-VALIDATION + chỉ số CÓ TRỌNG SỐ  &  #1 KIỂM TOÁN CÔNG BẰNG (fairness).

(#2) Thay vì 1 lần chia, đánh giá 5-fold: báo cáo AUC (thường) và AUC CÓ TRỌNG SỐ
     khảo sát (đại diện dân số), dạng mean ± std → kết quả ổn định, đáng tin.

(#1) Mô hình có công bằng giữa các nhóm không? Tách AUC / Brier (calibration) /
     độ nhạy / FPR / tỉ lệ gắn cờ theo: giới, chủng tộc, mức thu nhập, nhóm tuổi.
     Chênh lệch lớn = mô hình thiên lệch (vấn đề đạo đức + khoa học).

Đầu ra:
  outputs/figures/nhanes/fairness_auc.png
  outputs/figures/nhanes/fairness_threshold.png
  outputs/figures/nhanes/fairness_calibration.png
  outputs/metrics/fairness_metrics.csv, cv_metrics.csv

Chạy:  python3 scripts/fairness_nhanes.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"


def xgb(y):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=-1, random_state=C.RANDOM_STATE,
                         scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))


def grp_metrics(yt, pr, wt, thr):
    pred = (pr >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    auc = roc_auc_score(yt, pr, sample_weight=wt) if len(set(yt)) > 1 else np.nan
    return {"n": len(yt), "AUC_w": round(auc, 3), "Brier": round(brier_score_loss(yt, pr), 3),
            "sens": round(sens, 3), "FPR": round(fpr, 3), "PPV": round(ppv, 3),
            "flag": round(pred.mean(), 3)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pooled = nhanes.load_pooled()
    d = pooled[(pooled["age"] >= 18) & pooled["dep_risk"].notna()].reset_index(drop=True)
    X = d[nm.NUMERIC + nm.CATEGORICAL].copy()
    for c in nm.CATEGORICAL:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype("object")
    y = d["dep_risk"].astype(int)
    w = d["weight"].fillna(d["weight"].median())

    # ===== (#2) Cross-validation: AUC thường vs AUC có trọng số =====
    print("== (#2) 5-fold CV — AUC thường vs CÓ TRỌNG SỐ ==")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    au, auw = [], []
    for tr, va in skf.split(X, y):
        pipe = Pipeline([("prep", nm.build_preprocessor()), ("clf", xgb(y.iloc[tr]))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        au.append(roc_auc_score(y.iloc[va], p))
        auw.append(roc_auc_score(y.iloc[va], p, sample_weight=w.iloc[va]))
    print(f"  AUC (thường)     {np.mean(au):.3f} ± {np.std(au):.3f}")
    print(f"  AUC (trọng số)   {np.mean(auw):.3f} ± {np.std(auw):.3f}")
    pd.DataFrame({"fold": range(1, 6), "auc": np.round(au, 4),
                  "auc_weighted": np.round(auw, 4)}).to_csv(
        C.METRIC_DIR / "cv_metrics.csv", index=False)

    # ===== (#1) Fairness: 1 mô hình, đánh giá tách theo nhóm =====
    print("\n== (#1) Kiểm toán công bằng theo nhóm ==")
    Xtr, Xtmp, ytr, ytmp, wtr, wtmp, dtr, dtmp = train_test_split(
        X, y, w, d, test_size=0.4, stratify=y, random_state=C.RANDOM_STATE)
    Xva, Xte, yva, yte, wva, wte, dva, dte = train_test_split(
        Xtmp, ytmp, wtmp, dtmp, test_size=0.5, stratify=ytmp, random_state=C.RANDOM_STATE)

    cal = CalibratedClassifierCV(
        Pipeline([("prep", nm.build_preprocessor()), ("clf", xgb(ytr))]),
        method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    # ngưỡng sàng lọc: recall>=0.80 trên valid
    pva = cal.predict_proba(Xva)[:, 1]
    o = np.argsort(pva)[::-1]
    thr = float(pva[o][min(np.searchsorted(np.cumsum(yva.values[o]) / yva.sum(), 0.80),
                           len(o) - 1)])
    pte = cal.predict_proba(Xte)[:, 1]
    print(f"  Ngưỡng sàng lọc dùng chung = {thr:.3f}\n")

    # định nghĩa các nhóm trên tập test
    inc = pd.cut(dte["income_poverty"], nhanes.INCOME_BINS, labels=nhanes.INCOME_LABELS, right=False)
    groups = {
        "Giới": dte["gender"],
        "Chủng tộc": dte["race"].map(nhanes.RACE_LABELS),
        "Thu nhập": inc,
        "Nhóm tuổi": dte["age_group"],
    }
    rows = []
    for var, series in groups.items():
        for g in series.dropna().unique():
            m = (series == g).values
            if m.sum() < 50:
                continue
            r = grp_metrics(yte[m], pte[m], wte[m], thr)
            rows.append({"biến": var, "nhóm": str(g), **r})
    tab = pd.DataFrame(rows)
    tab.to_csv(C.METRIC_DIR / "fairness_metrics.csv", index=False)
    print(tab.to_string(index=False))

    # tóm tắt chênh lệch (gap) theo từng biến
    print("\n  Chênh lệch (max-min) giữa các nhóm:")
    for var in groups:
        sub = tab[tab["biến"] == var]
        print(f"    {var:<10} ΔAUC={sub.AUC_w.max()-sub.AUC_w.min():.3f}  "
              f"Δsens={sub.sens.max()-sub.sens.min():.3f}  "
              f"Δflag={sub.flag.max()-sub.flag.min():.3f}")

    _plots(tab, groups, dte, yte, pte)


def _plots(tab, groups, dte, yte, pte):
    palette = {"Giới": "#4f81bd", "Chủng tộc": "#9bbb59", "Thu nhập": "#c0504d", "Nhóm tuổi": "#8064a2"}
    # AUC theo nhóm
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(tab)), tab["AUC_w"], color=[palette[v] for v in tab["biến"]])
    ax.set_xticks(range(len(tab))); ax.set_xticklabels(tab["nhóm"], rotation=40, ha="right", fontsize=8)
    ax.axhline(tab["AUC_w"].mean(), ls="--", color="grey", label="TB chung")
    ax.set_ylim(0.6, 0.9); ax.set_ylabel("AUC có trọng số")
    ax.set_title("Công bằng: AUC theo từng nhóm (màu = biến phân nhóm)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fairness_auc.png", dpi=130); plt.close(fig)
    print("  + fairness_auc.png")

    # sens & FPR & flag theo nhóm
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(tab)); wd = 0.27
    ax.bar(x - wd, tab["sens"], wd, label="Độ nhạy (sens)", color="#2c7fb8")
    ax.bar(x, tab["FPR"], wd, label="FPR", color="#d95f0e")
    ax.bar(x + wd, tab["flag"], wd, label="Tỉ lệ gắn cờ", color="#9bbb59")
    ax.set_xticks(x); ax.set_xticklabels(tab["nhóm"], rotation=40, ha="right", fontsize=8)
    ax.set_title("Công bằng tại ngưỡng sàng lọc chung — chênh lệch giữa các nhóm")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fairness_threshold.png", dpi=130); plt.close(fig)
    print("  + fairness_threshold.png")

    # calibration theo mức thu nhập
    inc = pd.cut(dte["income_poverty"], nhanes.INCOME_BINS, labels=nhanes.INCOME_LABELS, right=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="hoàn hảo")
    for lab in nhanes.INCOME_LABELS:
        m = (inc == lab).values
        if m.sum() < 100:
            continue
        frac, mean = calibration_curve(yte[m], pte[m], n_bins=5, strategy="quantile")
        ax.plot(mean, frac, "o-", label=lab)
    ax.set_xlabel("Xác suất dự đoán TB"); ax.set_ylabel("Tỉ lệ dương thực tế")
    ax.set_title("Calibration theo mức thu nhập"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fairness_calibration.png", dpi=130); plt.close(fig)
    print("  + fairness_calibration.png, fairness_metrics.csv, cv_metrics.csv")


if __name__ == "__main__":
    main()
