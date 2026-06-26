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
from src import nhanes
from src.nhanes import wprev, wprev_by

FIG = C.FIG_DIR / "nhanes"
CYC_ORDER = [v[1] for v in nhanes.CYCLES.values()]
CYC_YEAR = {v[1]: v[0] for v in nhanes.CYCLES.values()}
AGE_ORDER = ["18-25", "26-35", "36-50", "51-65", "66+"]


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)
    print(f"  + {name}")


def _bar(ax, series, title, color):
    series = series.dropna()
    ax.bar(range(len(series)), series.values, color=color)
    ax.set_xticks(range(len(series))); ax.set_xticklabels(series.index, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel("% trầm cảm", fontsize=8)


def plot_factor_prevalence(ad):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    # giới
    _bar(axes[0, 0], wprev_by(ad, "gender", ["Male", "Female"]), "Giới tính", "#4f81bd")
    # chủng tộc
    r = ad.assign(rl=ad["race"].map(nhanes.RACE_LABELS))
    _bar(axes[0, 1], wprev_by(r, "rl", list(nhanes.RACE_LABELS.values())), "Chủng tộc", "#9bbb59")
    # hôn nhân
    _bar(axes[0, 2], wprev_by(ad, "marital", ["partnered", "previously", "never"]),
         "Tình trạng hôn nhân", "#8064a2")
    # thu nhập
    inc = ad.assign(ib=pd.cut(ad["income_poverty"], nhanes.INCOME_BINS,
                              labels=nhanes.INCOME_LABELS, right=False))
    _bar(axes[0, 3], wprev_by(inc, "ib", nhanes.INCOME_LABELS), "Mức thu nhập", "#c0504d")
    # sức khoẻ tự đánh giá
    g = ad.assign(gl=ad["gen_health"].map(nhanes.GENHEALTH_LABELS))
    _bar(axes[1, 0], wprev_by(g, "gl", list(nhanes.GENHEALTH_LABELS.values())),
         "Sức khoẻ tự đánh giá", "#e8a33d")
    # hút thuốc
    s = ad.assign(sl=ad["smoke_status"].map(nhanes.SMOKE_LABELS))
    _bar(axes[1, 1], wprev_by(s, "sl", list(nhanes.SMOKE_LABELS.values())), "Hút thuốc", "#d95f0e")
    # vận động vừa
    p = ad.assign(pl=ad["phys_moderate"].map({1: "Có", 2: "Không"}))
    _bar(axes[1, 2], wprev_by(p, "pl", ["Có", "Không"]), "Vận động vừa hằng tuần", "#41b6c4")
    # giấc ngủ
    sl = ad.dropna(subset=["sleep_hours"]).copy()
    sl["g"] = pd.cut(sl["sleep_hours"], [0, 5.9, 6.9, 8.9, 24], labels=["<6h", "6-7h", "7-9h", ">9h"])
    _bar(axes[1, 3], wprev_by(sl, "g", ["<6h", "6-7h", "7-9h", ">9h"]), "Thời lượng ngủ", "#2c7fb8")
    fig.suptitle("Tỉ lệ trầm cảm theo nhiều yếu tố — người lớn 18+, NHANES 2007–2023 (có trọng số)",
                 fontsize=13)
    _save(fig, "factor_prevalence.png")


def _trend_lines(df, col, order, title, fname, colors, labeller=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [CYC_YEAR[c] for c in CYC_ORDER]
    for g, color in zip(order, colors):
        ys = [wprev(df[(df["cycle"] == c) & (df[col] == g)]) for c in CYC_ORDER]
        ax.plot(xs, ys, "o-", color=color, label=labeller(g) if labeller else g)
    ax.set_xlabel("Năm bắt đầu chu kỳ"); ax.set_ylabel("% trầm cảm (PHQ-9 ≥ 10)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, fname)


def plot_trend_income(ad):
    inc = ad.assign(ib=pd.cut(ad["income_poverty"], nhanes.INCOME_BINS,
                              labels=nhanes.INCOME_LABELS, right=False))
    _trend_lines(inc, "ib", nhanes.INCOME_LABELS, "Trầm cảm theo mức thu nhập qua thời gian",
                 "trend_by_income.png", ["#d7301f", "#fc8d59", "#74a9cf", "#0570b0"])


def plot_trend_race(ad):
    _trend_lines(ad, "race", [1, 2, 3, 4, 5], "Trầm cảm theo chủng tộc qua thời gian",
                 "trend_by_race.png", ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"],
                 labeller=lambda g: nhanes.RACE_LABELS[g])


def plot_trend_gender_age(ad):
    """Giới × nhóm tuổi: kiểm tra phụ nữ trẻ có tăng mạnh nhất không."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    xs = [CYC_YEAR[c] for c in CYC_ORDER]
    colors = ["#2c7fb8", "#41b6c4", "#a1dab4", "#fec44f", "#d95f0e"]
    for ax, gender in zip(axes, ["Female", "Male"]):
        sub = ad[ad["gender"] == gender]
        for ag, col in zip(AGE_ORDER, colors):
            ys = [wprev(sub[(sub["cycle"] == c) & (sub["age_group"] == ag)]) for c in CYC_ORDER]
            ax.plot(xs, ys, "o-", color=col, label=ag)
        ax.set_title(gender); ax.set_xlabel("Năm"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("% trầm cảm"); axes[0].legend(title="Nhóm tuổi", fontsize=8)
    fig.suptitle("Trầm cảm theo Giới × Nhóm tuổi qua thời gian")
    _save(fig, "trend_by_gender_age.png")


def main():
    df = nhanes.load_pooled()
    ad = df[(df["age"] >= 18) & df["dep_risk"].notna()].copy()
    print(f"Phân tích {len(ad):,} người lớn")
    plot_factor_prevalence(ad)
    plot_trend_income(ad)
    plot_trend_race(ad)
    plot_trend_gender_age(ad)
    print("Xong.")


if __name__ == "__main__":
    main()
