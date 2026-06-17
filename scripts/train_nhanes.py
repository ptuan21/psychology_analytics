"""
Huấn luyện mô hình dự đoán nguy cơ trầm cảm (NHANES thật), chia TRAIN/TEST.

Hai cách đánh giá:
  (1) Split ngẫu nhiên phân tầng 80/20  -> năng lực dự đoán tổng quát.
  (2) Split THEO THỜI GIAN: học 2007–2018, kiểm 2021–2023
      -> mô hình học từ quá khứ có dự báo được dân số hậu đại dịch không?

Đầu ra:
  outputs/figures/nhanes/roc_nhanes.png        ROC các model
  outputs/figures/nhanes/cm_nhanes.png         confusion matrix (model tốt nhất)
  outputs/figures/nhanes/importance_nhanes.png top đặc trưng
  outputs/metrics/metrics_nhanes.csv           bảng metric
  outputs/models/model_nhanes_deprisk.joblib   model tốt nhất (đã .gitignore)

Chạy:  python3 scripts/train_nhanes.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import evaluate, nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"
LABELS = ["Không", "Nguy cơ"]


def evaluate_split(X_tr, X_te, y_tr, y_te, tag):
    rows, fitted, curves = [], {}, {}
    for name, est in nm.make_models().items():
        pipe = nm.fit_pipeline(name, est, X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1] if name != "dummy" else None
        s = evaluate.scores_clf(y_te, pred, proba)
        if proba is not None:
            s["pr_auc"] = round(average_precision_score(y_te, proba), 4)
            curves[name] = (y_te.values, proba)
        s["model"] = name
        rows.append(s); fitted[name] = (pipe, pred, proba)
        extra = "  ".join(f"{k}={v}" for k, v in s.items() if k != "model")
        print(f"  {name:<14} {extra}")
    res = pd.DataFrame(rows).set_index("model")
    best = res["roc_auc"].idxmax()
    print(f"  -> tốt nhất ({tag}): {best}  (AUC={res.loc[best,'roc_auc']})")
    return res, fitted, curves, best


def main():
    for d in (FIG, C.MODEL_DIR, C.METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = nhanes.load_pooled()
    X, y = nm.select_xy(df)
    years = df[(df["age"] >= 18) & df[nm.TARGET].notna()]["year"].values
    print(f"Dữ liệu: {len(X):,} người lớn 18+ | tỉ lệ dương (trầm cảm) = {y.mean():.1%}\n")

    # (1) split ngẫu nhiên phân tầng
    print("== (1) Split ngẫu nhiên phân tầng 80/20 ==")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
    res, fitted, curves, best = evaluate_split(Xtr, Xte, ytr, yte, "random")

    evaluate.save_roc(curves, "ROC — dự đoán nguy cơ trầm cảm (NHANES)", FIG / "roc_nhanes.png")
    pipe, pred, _ = fitted[best]
    evaluate.save_confusion(yte, pred, LABELS, f"NHANES — {best}", FIG / "cm_nhanes.png")
    evaluate.save_feature_importance_named(
        pipe.named_steps["clf"], nm.feature_names(pipe.named_steps["prep"]),
        "trầm cảm (NHANES)", FIG / "importance_nhanes.png")
    res.to_csv(C.METRIC_DIR / "metrics_nhanes.csv")
    joblib.dump({"pipeline": pipe, "labels": LABELS},
                C.MODEL_DIR / "model_nhanes_deprisk.joblib", compress=3)
    print("  + roc_nhanes.png, cm_nhanes.png, importance_nhanes.png, metrics_nhanes.csv\n")

    # (2) split theo thời gian: học quá khứ -> kiểm hiện tại
    print("== (2) Split theo thời gian: train ≤2018, test 2021–2023 ==")
    tr, te = years <= 2018, years >= 2021
    _, fit2, curves2, best2 = evaluate_split(X[tr], X[te], y[tr], y[te], "temporal")
    evaluate.save_roc(curves2, "ROC — học quá khứ, dự báo hiện tại (NHANES)",
                      FIG / "roc_nhanes_temporal.png")
    print("  + roc_nhanes_temporal.png")
    print(f"\nGhi chú: AUC ngẫu nhiên={res.loc[best,'roc_auc']:.3f} vs "
          f"theo thời gian={_auc(fit2, X[te], y[te]):.3f} "
          "(chênh lệch cho thấy mức 'lệch phân phối' theo thời gian).")


def _auc(fitted, X_te, y_te):
    best = max((n for n in fitted if n != "dummy"),
               key=lambda n: roc_auc_score(y_te, fitted[n][2]))
    return roc_auc_score(y_te, fitted[best][2])


if __name__ == "__main__":
    main()
