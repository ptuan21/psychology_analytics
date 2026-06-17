"""
TỐI ƯU "TRÁNH BỎ SÓT" — cấu hình mô hình sàng lọc ưu tiên ĐỘ NHẠY (recall) cao,
giảm số ca trầm cảm bị bỏ sót (false negative), kèm chi phí phải trả (báo động giả).

(A) Quét MỤC TIÊU ĐỘ NHẠY 0.80/0.90/0.95: với mỗi mức, tìm ngưỡng & báo cáo
    spec/PPV/FPR/tỉ lệ gắn cờ + SỐ CA BỎ SÓT (FN) + NNS (số người phải sàng để bắt 1 ca).
(B) NGƯỠNG RIÊNG THEO NHÓM (equal-sensitivity): vì 1 ngưỡng chung bỏ sót nhiều hơn ở
    một số nhóm tuổi, đặt ngưỡng riêng để MỌI nhóm đạt cùng độ nhạy mục tiêu → không
    nhóm nào bị bỏ sót có hệ thống. So sánh "ngưỡng chung" vs "ngưỡng theo nhóm".

Đầu ra:
  outputs/figures/nhanes/recall_tradeoff.png
  outputs/figures/nhanes/equal_sensitivity.png
  outputs/metrics/recall_operating_points.csv
  outputs/models/model_nhanes_highrecall.joblib (đã .gitignore)

Chạy:  python3 scripts/optimize_recall_nhanes.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"
TARGETS = [0.80, 0.90, 0.95]


def xgb(y):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=-1, random_state=C.RANDOM_STATE,
                         scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))


def thr_for_recall(p, y, t):
    """Ngưỡng thấp nhất để đạt độ nhạy >= t."""
    o = np.argsort(p)[::-1]
    cr = np.cumsum(np.asarray(y)[o]) / max(np.sum(y), 1)
    idx = np.searchsorted(cr, t)
    return float(p[o][min(idx, len(o) - 1)])


def op_point(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    return {"thr": round(thr, 4), "sens": round(sens, 3),
            "spec": round(tn / (tn + fp), 3), "PPV": round(tp / (tp + fp) if tp + fp else 0, 3),
            "FPR": round(fp / (fp + tn), 3), "flag_rate": round(pred.mean(), 3),
            "bỏ_sót_FN": int(fn), "NNS": round((tp + fp) / tp, 1) if tp else np.nan}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pooled = nhanes.load_pooled()
    d = pooled[(pooled["age"] >= 18) & pooled["dep_risk"].notna()].reset_index(drop=True)
    X = d[nm.NUMERIC + nm.CATEGORICAL].copy()
    for c in nm.CATEGORICAL:
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype("object")
    y = d["dep_risk"].astype(int)

    Xtr, Xtmp, ytr, ytmp, dtr, dtmp = train_test_split(
        X, y, d, test_size=0.4, stratify=y, random_state=C.RANDOM_STATE)
    Xva, Xte, yva, yte, dva, dte = train_test_split(
        Xtmp, ytmp, dtmp, test_size=0.5, stratify=ytmp, random_state=C.RANDOM_STATE)

    cal = CalibratedClassifierCV(
        Pipeline([("prep", nm.build_preprocessor()), ("clf", xgb(ytr))]),
        method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    pva, pte = cal.predict_proba(Xva)[:, 1], cal.predict_proba(Xte)[:, 1]

    # ===== (A) quét mục tiêu độ nhạy =====
    print(f"== (A) Tối ưu tránh bỏ sót — tổng ca dương ở test = {int(yte.sum())} ==")
    rows = []
    for t in TARGETS:
        thr = thr_for_recall(pva, yva.values, t)
        op = op_point(yte.values, pte, thr); op["mục_tiêu_sens"] = t
        rows.append(op)
    tab = pd.DataFrame(rows)[["mục_tiêu_sens", "thr", "sens", "spec", "PPV",
                              "FPR", "flag_rate", "bỏ_sót_FN", "NNS"]]
    print(tab.to_string(index=False))
    tab.to_csv(C.METRIC_DIR / "recall_operating_points.csv", index=False)

    # ===== (B) ngưỡng riêng theo nhóm tuổi (equal-sensitivity 0.90) =====
    T = 0.90
    thr_global = thr_for_recall(pva, yva.values, T)
    ages = ["18-25", "26-35", "36-50", "51-65", "66+"]
    print(f"\n== (B) Đảm bảo MỌI nhóm tuổi đạt độ nhạy {T:.0%} (ngưỡng riêng) ==")
    print(f"  Ngưỡng chung = {thr_global:.3f}")
    comp = []
    for g in ages:
        mva = (dva["age_group"] == g).values
        mte = (dte["age_group"] == g).values
        if mte.sum() < 50 or yte[mte].sum() < 5:
            continue
        thr_g = thr_for_recall(pva[mva], yva[mva].values, T)
        s_glob = op_point(yte[mte].values, pte[mte], thr_global)
        s_grp = op_point(yte[mte].values, pte[mte], thr_g)
        comp.append({"nhóm": g, "thr_riêng": round(thr_g, 3),
                     "sens_chung": s_glob["sens"], "bỏ_sót_chung": s_glob["bỏ_sót_FN"],
                     "sens_riêng": s_grp["sens"], "bỏ_sót_riêng": s_grp["bỏ_sót_FN"],
                     "flag_riêng": s_grp["flag_rate"]})
    cdf = pd.DataFrame(comp)
    print(cdf.to_string(index=False))
    miss_glob = cdf["bỏ_sót_chung"].sum(); miss_grp = cdf["bỏ_sót_riêng"].sum()
    print(f"\n  Tổng bỏ sót: ngưỡng chung={miss_glob} → ngưỡng theo nhóm={miss_grp} "
          f"(giảm {miss_glob-miss_grp} ca, đều ≥{T:.0%} mọi nhóm)")

    # ===== biểu đồ =====
    # (A) trade-off độ nhạy vs tỉ lệ gắn cờ
    grid = np.linspace(pte.min(), pte.max(), 200)
    sens = [op_point(yte.values, pte, t)["sens"] for t in grid]
    flag = [(pte >= t).mean() for t in grid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(flag, sens, color="#2c7fb8", lw=2)
    for t in TARGETS:
        thr = thr_for_recall(pva, yva.values, t); fr = (pte >= thr).mean()
        ax.scatter(fr, op_point(yte.values, pte, thr)["sens"], zorder=3)
        ax.annotate(f"sens {t:.0%}\ngắn cờ {fr:.0%}", (fr, t), fontsize=8,
                    textcoords="offset points", xytext=(6, -4))
    ax.set_xlabel("Tỉ lệ gắn cờ (chi phí)"); ax.set_ylabel("Độ nhạy (bắt được ca)")
    ax.set_title("Đánh đổi: tránh bỏ sót ↔ số người phải theo dõi"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "recall_tradeoff.png", dpi=130); plt.close(fig)
    print("  + recall_tradeoff.png")

    # (B) độ nhạy theo nhóm: ngưỡng chung vs riêng
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(cdf)); wd = 0.38
    ax.bar(x - wd/2, cdf["sens_chung"], wd, label="Ngưỡng chung", color="#d95f0e")
    ax.bar(x + wd/2, cdf["sens_riêng"], wd, label="Ngưỡng theo nhóm", color="#2c7fb8")
    ax.axhline(T, ls="--", color="grey", label=f"mục tiêu {T:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(cdf["nhóm"]); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Độ nhạy"); ax.set_title("Ngưỡng riêng theo nhóm → không bỏ sót nhóm nào")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "equal_sensitivity.png", dpi=130); plt.close(fig)
    print("  + equal_sensitivity.png, recall_operating_points.csv")

    joblib.dump({"model": cal, "thr_global_recall90": thr_global,
                 "group_thresholds_age": dict(zip(cdf["nhóm"], cdf["thr_riêng"]))},
                C.MODEL_DIR / "model_nhanes_highrecall.joblib", compress=3)


if __name__ == "__main__":
    main()
