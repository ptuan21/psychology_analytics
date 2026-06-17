"""
MÔ HÌNH THEO NHÓM YẾU TỐ — đo sức mạnh dự đoán trầm cảm của từng NHÓM yếu tố
xã hội & con người, và tinh chỉnh bằng ENSEMBLE.

Bốn nhóm yếu tố:
  DEMOGRAPHIC  tuổi, giới, chủng tộc            (bối cảnh, không can thiệp được)
  SOCIAL       học vấn, thu nhập, hôn nhân, an ninh lương thực, việc làm, bảo hiểm
  HUMAN        ngủ, ít vận động, hút thuốc      (lối sống — can thiệp được)
  HEALTH       sức khỏe tự đánh giá             (trạng thái hiện tại; lưu ý: gần hệ quả)

Phân tích:
  (a) STANDALONE — mỗi nhóm DỰ ĐOÁN RIÊNG đạt AUC bao nhiêu.
  (b) INCREMENTAL — cộng dồn Demographic→Social→Human→Health, ΔAUC = đóng góp riêng.
  (c) ENSEMBLE — stacking (LogReg+RF+XGB) trên toàn bộ, so với XGB đơn.

Đầu ra:
  outputs/figures/nhanes/factor_domain_standalone.png
  outputs/figures/nhanes/factor_domain_incremental.png
  outputs/metrics/factor_domains.csv

Chạy:  python3 scripts/factor_models_nhanes.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"

# (numeric, categorical) cho mỗi nhóm
DOMAINS = {
    "Demographic": (["age"], ["gender", "race"]),
    "Social":      (["income_poverty", "food_security"],
                    ["education_label", "marital", "employment", "insured"]),
    "Human/lối sống": (["sleep_hours", "sedentary_min"], ["smoke_status"]),
    "Health":      (["gen_health"], []),
}
ORDER = ["Demographic", "Social", "Human/lối sống", "Health"]


def make_prep(num, cat):
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), cat))
    return ColumnTransformer(parts)


def xgb(y_tr):
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         n_jobs=-1, random_state=C.RANDOM_STATE,
                         scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1))


def auc_of(num, cat, Xtr, Xte, ytr, yte):
    pipe = Pipeline([("prep", make_prep(num, cat)), ("clf", xgb(ytr))])
    pipe.fit(Xtr[num + cat], ytr)
    p = pipe.predict_proba(Xte[num + cat])[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    X, y = nm.select_xy(nhanes.load_pooled())
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
    print(f"Train={len(Xtr):,}  Test={len(Xte):,}\n")

    rows = []
    # (a) standalone
    print("== (a) Mỗi nhóm dự đoán RIÊNG ==")
    for d in ORDER:
        num, cat = DOMAINS[d]
        a, pr = auc_of(num, cat, Xtr, Xte, ytr, yte)
        rows.append({"nhóm": d, "loại": "standalone", "AUC": round(a, 4), "PR_AUC": round(pr, 4)})
        print(f"  {d:<16} AUC={a:.3f}  PR-AUC={pr:.3f}")

    # (b) incremental
    print("\n== (b) Cộng dồn (ΔAUC = đóng góp riêng) ==")
    num_c, cat_c, prev = [], [], 0.5
    for d in ORDER:
        num_c += DOMAINS[d][0]; cat_c += DOMAINS[d][1]
        a, pr = auc_of(num_c, cat_c, Xtr, Xte, ytr, yte)
        rows.append({"nhóm": f"+{d}", "loại": "cumulative", "AUC": round(a, 4),
                     "PR_AUC": round(pr, 4), "dAUC": round(a - prev, 4)})
        print(f"  +{d:<15} AUC={a:.3f}  (Δ {a-prev:+.3f})")
        prev = a
    full_auc = prev

    # (c) ensemble (stacking) trên toàn bộ đặc trưng
    print("\n== (c) Ensemble (stacking) vs XGB đơn ==")
    num_all, cat_all = nm.NUMERIC, nm.CATEGORICAL
    prep = nm.build_preprocessor()
    estimators = [
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                      class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE)),
        ("xgb", xgb(ytr)),
    ]
    stack = Pipeline([("prep", prep), ("clf", StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(max_iter=1000),
        cv=3, n_jobs=-1))])
    stack.fit(Xtr, ytr)
    ps = stack.predict_proba(Xte)[:, 1]
    a_stack = roc_auc_score(yte, ps)
    rows.append({"nhóm": "Ensemble (stacking)", "loại": "ensemble",
                 "AUC": round(a_stack, 4), "PR_AUC": round(average_precision_score(yte, ps), 4)})
    print(f"  XGB đơn (full)      AUC={full_auc:.3f}")
    print(f"  Stacking ensemble   AUC={a_stack:.3f}  (Δ {a_stack-full_auc:+.3f})")

    pd.DataFrame(rows).to_csv(C.METRIC_DIR / "factor_domains.csv", index=False)

    # ---- biểu đồ ----
    stand = [r for r in rows if r["loại"] == "standalone"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([r["nhóm"] for r in stand], [r["AUC"] for r in stand],
                  color=["#8064a2", "#c0504d", "#4f81bd", "#e8a33d"])
    ax.axhline(0.5, ls="--", color="grey"); ax.set_ylim(0.5, 0.8)
    ax.set_ylabel("AUC (dự đoán riêng)"); ax.set_title("Sức mạnh dự đoán RIÊNG của từng nhóm yếu tố")
    for b, r in zip(bars, stand):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{r['AUC']:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "factor_domain_standalone.png", dpi=130); plt.close(fig)
    print("  + factor_domain_standalone.png")

    cum = [r for r in rows if r["loại"] == "cumulative"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(range(len(cum)), [r["AUC"] for r in cum], "o-", color="#2c7fb8", lw=2)
    ax.set_xticks(range(len(cum))); ax.set_xticklabels([r["nhóm"] for r in cum], rotation=15)
    for i, r in enumerate(cum):
        ax.annotate(f"Δ{r['dAUC']:+.3f}", (i, r["AUC"]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_ylabel("AUC cộng dồn"); ax.set_title("Đóng góp riêng của từng nhóm (thêm dần)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "factor_domain_incremental.png", dpi=130); plt.close(fig)
    print("  + factor_domain_incremental.png, factor_domains.csv")


if __name__ == "__main__":
    main()
