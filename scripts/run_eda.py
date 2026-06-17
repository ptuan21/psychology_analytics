"""
Trực quan hoá dữ liệu (EDA) — sinh biểu đồ vào outputs/figures/eda/

  1. eda_severity_distribution.png : phân bố 5 mức severity x 3 thang
  2. eda_score_histograms.png      : phân bố điểm liên tục 0–42 từng thang
  3. eda_correlation_heatmap.png   : tương quan giữa điểm DASS & Big Five
  4. eda_bigfive_vs_dass.png       : Big Five tương quan thế nào với mức độ nặng
  5. eda_demographics.png          : tuổi, giới, học vấn, ngôn ngữ mẹ đẻ
  6. eda_score_by_group.png        : điểm trung bình theo giới & nhóm tuổi

Chạy:  python3 scripts/run_eda.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import data_loader, scoring

EDA_DIR = C.FIG_DIR / "eda"
SUBS = list(C.DASS_SUBSCALES)              # ['depression','anxiety','stress']
COLORS = {"depression": "#c0504d", "anxiety": "#4f81bd", "stress": "#9bbb59"}
BIG5 = [f"big5_{k}" for k in C.BIG_FIVE]


def _save(fig, name):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(EDA_DIR / name, dpi=130); plt.close(fig)
    print(f"  + {name}")


def plot_severity_distribution(df):
    levels = C.SEVERITY_ORDER
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    x = np.arange(len(levels))
    for i, s in enumerate(SUBS):
        counts = df[f"{s}_level"].value_counts().reindex(levels).fillna(0)
        pct = counts / counts.sum() * 100
        ax.bar(x + (i - 1) * w, pct, w, label=s, color=COLORS[s])
    ax.set_xticks(x); ax.set_xticklabels(levels, rotation=20)
    ax.set_ylabel("% người tham gia"); ax.set_title("Phân bố mức độ severity theo thang")
    ax.legend()
    _save(fig, "eda_severity_distribution.png")


def plot_score_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SUBS):
        ax.hist(df[f"{s}_score"], bins=range(0, 44, 2), color=COLORS[s], edgecolor="white")
        ax.axvline(C.HIGH_RISK_CUTOFF[s], color="black", ls="--",
                   label=f"ngưỡng nguy cơ ={C.HIGH_RISK_CUTOFF[s]}")
        ax.set_title(s); ax.set_xlabel("điểm 0–42"); ax.legend(fontsize=8)
    axes[0].set_ylabel("số người")
    fig.suptitle("Phân bố điểm DASS liên tục")
    _save(fig, "eda_score_histograms.png")


def plot_correlation_heatmap(df):
    cols = [f"{s}_score" for s in SUBS] + BIG5
    short = [s[:4] for s in SUBS] + [k[:5] for k in C.BIG_FIVE]
    corr = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax.set_title("Tương quan: điểm DASS ↔ Big Five")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, "eda_correlation_heatmap.png")


def plot_bigfive_vs_dass(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(C.BIG_FIVE)); w = 0.25
    for i, s in enumerate(SUBS):
        corrs = [df[f"{s}_score"].corr(df[b]) for b in BIG5]
        ax.bar(x + (i - 1) * w, corrs, w, label=s, color=COLORS[s])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(list(C.BIG_FIVE), rotation=20)
    ax.set_ylabel("hệ số tương quan với điểm DASS")
    ax.set_title("Big Five dự báo mức độ nặng thế nào (âm = bảo vệ)")
    ax.legend()
    _save(fig, "eda_bigfive_vs_dass.png")


def plot_demographics(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # tuổi
    axes[0, 0].hist(df["age"], bins=range(13, 80, 2), color="#4f81bd", edgecolor="white")
    axes[0, 0].set_title("Tuổi"); axes[0, 0].set_xlabel("tuổi")
    # giới
    g = df["gender"].map({1: "Nam", 2: "Nữ", 3: "Khác"}).value_counts()
    axes[0, 1].bar(g.index, g.values, color="#9bbb59"); axes[0, 1].set_title("Giới tính")
    # học vấn
    e = df["education"].map({1: "<PT", 2: "PT", 3: "ĐH", 4: "Sau ĐH"}).value_counts()
    axes[1, 0].bar(e.index, e.values, color="#c0504d"); axes[1, 0].set_title("Học vấn")
    # tiếng Anh mẹ đẻ
    n = df["engnat"].map({1: "Có", 2: "Không"}).value_counts()
    axes[1, 1].bar(n.index, n.values, color="#8064a2"); axes[1, 1].set_title("Tiếng Anh là mẹ đẻ?")
    fig.suptitle("Nhân khẩu học người tham gia")
    _save(fig, "eda_demographics.png")


def plot_score_by_group(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # theo giới
    gmap = {1: "Nam", 2: "Nữ", 3: "Khác"}
    sub = df[df["gender"].isin(gmap)]
    means = sub.groupby(sub["gender"].map(gmap))[[f"{s}_score" for s in SUBS]].mean()
    means = means.reindex(["Nam", "Nữ", "Khác"])
    x = np.arange(len(means)); w = 0.25
    for i, s in enumerate(SUBS):
        axes[0].bar(x + (i - 1) * w, means[f"{s}_score"], w, label=s, color=COLORS[s])
    axes[0].set_xticks(x); axes[0].set_xticklabels(means.index)
    axes[0].set_ylabel("điểm TB"); axes[0].set_title("Điểm DASS theo giới"); axes[0].legend()
    # theo nhóm tuổi
    bins = [13, 18, 25, 35, 50, 100]; lab = ["13–17", "18–24", "25–34", "35–49", "50+"]
    grp = pd.cut(df["age"], bins=bins, labels=lab, right=False)
    means2 = df.groupby(grp, observed=True)[[f"{s}_score" for s in SUBS]].mean()
    for s in SUBS:
        axes[1].plot(means2.index.astype(str), means2[f"{s}_score"],
                     marker="o", label=s, color=COLORS[s])
    axes[1].set_ylabel("điểm TB"); axes[1].set_title("Điểm DASS theo nhóm tuổi"); axes[1].legend()
    _save(fig, "eda_score_by_group.png")


def main():
    print("Tải & tính điểm...")
    df = scoring.build_targets_and_features(data_loader.load_clean(verbose=False))
    print(f"Sinh biểu đồ EDA cho {len(df):,} người -> {EDA_DIR}/")
    plot_severity_distribution(df)
    plot_score_histograms(df)
    plot_correlation_heatmap(df)
    plot_bigfive_vs_dass(df)
    plot_demographics(df)
    plot_score_by_group(df)
    print("Xong.")


if __name__ == "__main__":
    main()
