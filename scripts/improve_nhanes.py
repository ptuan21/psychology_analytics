"""
CẢI TIẾN mô hình dự đoán trầm cảm (NHANES) — 3 bước, chống rò rỉ bằng tách 3 phần:
  TRAIN (60%): tinh chỉnh siêu tham số (RandomizedSearchCV) + hiệu chỉnh xác suất
  VALID (20%): CHỌN NGƯỠNG quyết định (cho sàng lọc)
  TEST  (20%): báo cáo cuối cùng (chưa từng đụng tới)

  (3) Tuning: RandomizedSearchCV cho RF & XGB, chọn model có CV-AUC cao nhất.
  (2) Calibration: CalibratedClassifierCV (isotonic) -> xác suất tin cậy (Brier thấp hơn).
  (1) Threshold: chọn ngưỡng cho sàng lọc (recall mục tiêu ~0.80) & Youden's J,
      ĐẶC TẢ như một test sàng lọc: độ nhạy/đặc hiệu/PPV/NPV/LR+/LR-.

Đây là công cụ SÀNG LỌC (không phải chẩn đoán): dự đoán nhãn PHQ-9 ≥ 10 — bản thân
PHQ-9 là thang sàng lọc đã kiểm định (Se≈0.85, Sp≈0.85 cho trầm cảm nặng so với phỏng
vấn lâm sàng; Levis et al. BMJ 2019). Vì vậy đầu ra cần hiểu là "khả năng sàng lọc dương",
dùng để PHÂN TẦNG & chuyển tuyến, không thay thế đánh giá của chuyên gia.

Chạy:  python3 scripts/improve_nhanes.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import evaluate, nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"


def tune(name, X, y):
    """RandomizedSearchCV trên pipeline (prep + clf). Trả về (best_pipe, cv_auc)."""
    base = nm.make_models()[name]
    if name == "xgboost":
        base.set_params(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1))
    pipe = Pipeline([("prep", nm.build_preprocessor()), ("clf", base)])
    search = RandomizedSearchCV(
        pipe, nm.GRIDS[name], n_iter=15, scoring="roc_auc", cv=4,
        n_jobs=-1, random_state=C.RANDOM_STATE, refit=True)
    search.fit(X, y)
    print(f"  {name:<14} CV-AUC={search.best_score_:.4f}")
    return search.best_estimator_, search.best_score_


def metrics_at(y, proba, thr):
    """Chỉ số SÀNG LỌC đầy đủ tại một ngưỡng (đặc tả như một test sàng lọc)."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    sens = tp / (tp + fn) if tp + fn else 0          # = recall = độ nhạy
    spec = tn / (tn + fp) if tn + fp else 0          # độ đặc hiệu
    ppv = tp / (tp + fp) if tp + fp else 0           # = precision
    npv = tn / (tn + fn) if tn + fn else 0
    lr_plus = sens / (1 - spec) if spec < 1 else float("inf")
    lr_minus = (1 - sens) / spec if spec > 0 else float("inf")
    # net benefit (Decision Curve Analysis, Vickers & Elkin 2006) tại chính ngưỡng đang dùng
    net_benefit = tp / n - fp / n * (thr / (1 - thr)) if 0 < thr < 1 else float("nan")
    return {"thr": round(thr, 3), "sens(recall)": round(sens, 3),
            "spec": round(spec, 3), "PPV": round(ppv, 3), "NPV": round(npv, 3),
            "LR+": round(lr_plus, 2), "LR-": round(lr_minus, 2),
            "net_benefit": round(net_benefit, 4),
            "flag_rate": round(pred.mean(), 3)}


def main():
    for d in (FIG, C.MODEL_DIR, C.METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)
    X, y = nm.select_xy(nhanes.load_pooled())

    # tách 60/20/20
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=C.RANDOM_STATE)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=C.RANDOM_STATE)
    print(f"Train={len(X_tr):,}  Valid={len(X_va):,}  Test={len(X_te):,}\n")

    # (3) tuning
    print("== (3) Tinh chỉnh siêu tham số ==")
    cands = {n: tune(n, X_tr, y_tr) for n in nm.GRIDS}
    best_name = max(cands, key=lambda n: cands[n][1])
    best_pipe = cands[best_name][0]
    print(f"  -> chọn: {best_name}\n")

    # (2) calibration
    print("== (2) Hiệu chỉnh xác suất (isotonic) ==")
    cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    p_un = best_pipe.predict_proba(X_te)[:, 1]
    p_cal = cal.predict_proba(X_te)[:, 1]
    evaluate.save_calibration({"Chưa hiệu chỉnh": (y_te.values, p_un),
                               "Đã hiệu chỉnh": (y_te.values, p_cal)},
                              "Calibration — nguy cơ trầm cảm (test)", FIG / "calibration_nhanes.png")
    print(f"  AUC test={roc_auc_score(y_te, p_cal):.3f}  PR-AUC={average_precision_score(y_te, p_cal):.3f}")
    print("  + calibration_nhanes.png\n")

    # (1) chọn ngưỡng trên VALID (không đụng test)
    print("== (1) Chọn ngưỡng quyết định (trên valid) ==")
    p_va = cal.predict_proba(X_va)[:, 1]
    fpr, tpr, thr = roc_curve(y_va, p_va)
    thr_youden = float(thr[np.argmax(tpr - fpr)])              # cân bằng nhạy/đặc hiệu
    # ngưỡng đạt recall >= 0.80 (sàng lọc: ưu tiên bắt hết ca nguy cơ)
    order = np.argsort(p_va)[::-1]
    cum_recall = np.cumsum(y_va.values[order]) / y_va.sum()
    idx = np.searchsorted(cum_recall, 0.80)
    thr_recall = float(p_va[order][min(idx, len(order) - 1)])

    marks = {"Youden": thr_youden, "Recall≥0.80": thr_recall, "Mặc định 0.5": 0.5}
    evaluate.save_threshold_curve(y_te, p_cal, "Precision/Recall theo ngưỡng (test)",
                                  FIG / "threshold_nhanes.png", marks)
    rows = [{"name": k, **metrics_at(y_te, p_cal, t)} for k, t in marks.items()]
    tab = pd.DataFrame(rows).set_index("name")
    print(tab.to_string())
    tab.to_csv(C.METRIC_DIR / "nhanes_thresholds.csv")
    print("  + threshold_nhanes.png, nhanes_thresholds.csv")

    # Decision Curve Analysis: mô hình có ích lâm sàng hơn "gắn cờ tất cả"/"không gắn cờ ai"
    # không, ở MỌI ngưỡng — bổ sung cho AUC/calibration (đo phân biệt/hiệu chỉnh, không đo lợi ích ra quyết định).
    evaluate.save_decision_curve(y_te.values, p_cal, "Decision Curve — nguy cơ trầm cảm (test)",
                                 FIG / "decision_curve_nhanes.png")
    print("  + decision_curve_nhanes.png")

    joblib.dump({"model": cal, "thresholds": marks, "best_name": best_name},
                C.MODEL_DIR / "model_nhanes_calibrated.joblib", compress=3)
    print(f"\nLưu model đã hiệu chỉnh + ngưỡng -> outputs/models/ (gitignore)")


if __name__ == "__main__":
    main()
