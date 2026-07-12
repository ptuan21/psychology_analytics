"""
Đánh giá & trực quan hoá: bảng metric, confusion matrix, độ quan trọng đặc trưng.
"""
import json

import matplotlib
matplotlib.use("Agg")  # backend không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)

from . import config as C
from .features import feature_names

# Metric dùng để chọn model tốt nhất, theo từng chế độ.
SELECTION_METRIC = {"multiclass": "macro_f1", "binary": "roc_auc", "regression": "r2"}


def scores_clf(y_true, y_pred, proba=None) -> dict:
    """Metric phân loại; có roc_auc nếu là nhị phân và truyền proba."""
    s = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }
    if proba is not None:
        s["roc_auc"] = round(roc_auc_score(y_true, proba), 4)
    return s


def scores_reg(y_true, y_pred) -> dict:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "r2": round(r2_score(y_true, y_pred), 4),
        "spearman": round(spearmanr(y_pred, y_true).correlation, 4),
    }


def save_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_roc(curves: dict, title, path):
    """curves = {tên_model: (y_true, proba)} -> vẽ ROC nhiều model lên một biểu đồ."""
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (yt, pr) in curves.items():
        fpr, tpr, _ = roc_curve(yt, pr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(yt, pr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ngẫu nhiên")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_calibration(curves: dict, title, path):
    """curves = {tên: (y_true, proba)} -> reliability diagram + Brier score."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="hoàn hảo")
    for name, (yt, pr) in curves.items():
        frac, mean = calibration_curve(yt, pr, n_bins=10, strategy="quantile")
        b = brier_score_loss(yt, pr)
        ax.plot(mean, frac, "o-", label=f"{name} (Brier={b:.3f})")
    ax.set_xlabel("Xác suất dự đoán trung bình"); ax.set_ylabel("Tỉ lệ dương thực tế")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_threshold_curve(y_true, proba, title, path, marks=None):
    """Precision/Recall/F1 theo ngưỡng + đánh dấu các ngưỡng đã chọn."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, prec[:-1], label="Precision", color="#4f81bd")
    ax.plot(thr, rec[:-1], label="Recall", color="#c0504d")
    ax.plot(thr, f1[:-1], label="F1", color="#9bbb59")
    for name, t in (marks or {}).items():
        ax.axvline(t, ls="--", color="grey", lw=1)
        ax.text(t, 1.01, name, rotation=90, fontsize=7, va="bottom", ha="center")
    ax.set_xlabel("Ngưỡng xác suất"); ax.set_ylabel("Giá trị"); ax.set_title(title)
    ax.legend(loc="center left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def decision_curve_points(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Net benefit theo ngưỡng quyết định pt (Decision Curve Analysis, Vickers & Elkin 2006):
    net_benefit(pt) = TP/n - FP/n * pt/(1-pt). So với "sàng lọc tất cả" (net benefit của việc
    gắn cờ mọi người) và "không sàng lọc ai" (net benefit = 0, đường mốc)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    prevalence = y_true.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for pt in thresholds:
        pred = (proba >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        w = pt / (1 - pt)
        rows.append({
            "threshold": pt,
            "net_benefit_model": tp / n - fp / n * w,
            "net_benefit_treat_all": prevalence - (1 - prevalence) * w,
        })
    return pd.DataFrame(rows)


def save_decision_curve(y_true, proba, title, path, thresholds=None):
    """Vẽ Decision Curve Analysis: mô hình có ích lâm sàng hơn "gắn cờ tất cả"/"không gắn cờ ai"
    không, ở mỗi ngưỡng xác suất quyết định — đo giá trị lâm sàng, không chỉ độ phân biệt (AUC)."""
    dcp = decision_curve_points(y_true, proba, thresholds)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(dcp["threshold"], dcp["net_benefit_model"], lw=2.2, color="#3b6ea5", label="Mô hình")
    ax.plot(dcp["threshold"], dcp["net_benefit_treat_all"], lw=1.5, ls="--",
            color="#c0504d", label="Sàng lọc tất cả")
    ax.axhline(0, lw=1.5, ls="--", color="grey", label="Không sàng lọc ai")
    # Vùng nhìn chuẩn cho DCA: zoom quanh 0 — "sàng lọc tất cả" giảm rất nhanh và không còn
    # ý nghĩa lâm sàng ở ngưỡng cao, nên không để nó kéo giãn trục Y làm mất phần đáng đọc.
    ymax = max(dcp["net_benefit_model"].max(), dcp["net_benefit_treat_all"].max(), 0)
    ax.set_ylim(-0.02, ymax * 1.15 if ymax > 0 else 0.05)
    ax.set_xlabel("Ngưỡng xác suất quyết định (pt)"); ax.set_ylabel("Net benefit")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_regression_scatter(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.15, color="#3b6ea5")
    lim = [0, 42]
    ax.plot(lim, lim, "r--", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Điểm thực tế (0–42)"); ax.set_ylabel("Điểm dự đoán")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_feature_importance_named(clf, names, target, path, top=20):
    """Như save_feature_importance nhưng nhận sẵn danh sách tên đặc trưng."""
    if not hasattr(clf, "feature_importances_"):
        return
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(idx))[::-1], imp[idx], color="#3b6ea5")
    ax.set_yticks(range(len(idx))[::-1]); ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_title(f"Top {top} đặc trưng — {target}"); ax.set_xlabel("Độ quan trọng")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_feature_importance(pipe, target, path, top=20):
    """Vẽ top đặc trưng quan trọng (RF/XGB: feature_importances_)."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = feature_names(pipe.named_steps["prep"])
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(idx))[::-1], imp[idx], color="#3b6ea5")
    ax.set_yticks(range(len(idx))[::-1]); ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.set_title(f"Top {top} đặc trưng — {target}"); ax.set_xlabel("Độ quan trọng")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def text_report(y_true, y_pred, labels) -> str:
    return classification_report(
        y_true, y_pred, labels=range(len(labels)),
        target_names=labels, zero_division=0, digits=3,
    )


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
