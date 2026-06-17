"""
Đánh giá & trực quan hoá: bảng metric, confusion matrix, độ quan trọng đặc trưng.
"""
import json

import matplotlib
matplotlib.use("Agg")  # backend không cần màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

from . import config as C
from .features import feature_names


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
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
