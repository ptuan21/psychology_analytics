"""
Phân tích xu hướng trầm cảm (NHANES 2007–2023) theo THỜI GIAN × TUỔI × BẬC HỌC,
và liên hệ với các yếu tố CAN THIỆP ĐƯỢC. Tất cả ước lượng đều DÙNG TRỌNG SỐ khảo sát
(WTMEC2YR) để cho ra tỉ lệ đại diện dân số (không phải tỉ lệ thô của mẫu).
Đầu ra:
  outputs/figures/nhanes/trend_overall.png        xu hướng chung + theo giới
  outputs/figures/nhanes/trend_by_age.png         theo nhóm tuổi qua các chu kỳ
  outputs/figures/nhanes/trend_by_education.png    theo bậc học qua các chu kỳ
  outputs/figures/nhanes/heatmap_age_education.png tỉ lệ theo tuổi × bậc học
  outputs/figures/nhanes/delta_by_group.png        nhóm nào tăng nhanh nhất
  outputs/figures/nhanes/modifiable_factors.png    trầm cảm theo ngủ/hút thuốc/vận động
  outputs/metrics/nhanes_summary.txt               số liệu chính
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
from src import nhanes

FIG = C.FIG_DIR / "nhanes"
AGE_ORDER = ["18-25", "26-35", "36-50", "51-65", "66+"]
EDU_ORDER = ["<9th", "9-11th", "HS/GED", "Some college", "College grad"]
CYC_ORDER = [v[1] for v in nhanes.CYCLES.values()]
CYC_YEAR = {v[1]: v[0] for v in nhanes.CYCLES.values()}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)
    print(f"  + {name}")


def wprev(d):
    """Tỉ lệ trầm cảm có trọng số (%) — bỏ dòng thiếu trọng số/nhãn."""
    d = d.dropna(subset=["dep_risk", "weight"])
    if d["weight"].sum() == 0 or len(d) == 0:
        return np.nan
    return 100 * np.average(d["dep_risk"].astype(float), weights=d["weight"])


def wprev_by(df, col, order):
    """Tỉ lệ có trọng số theo từng giá trị của `col`, theo thứ tự `order`."""
    return pd.Series({k: wprev(df[df[col] == k]) for k in order})


def trend_table(df, group_col, order):
    """Ma trận (group x cycle) tỉ lệ có trọng số."""
    out = pd.DataFrame(index=order, columns=CYC_ORDER, dtype=float)
    for cyc in CYC_ORDER:
        sub = df[df["cycle"] == cyc]
        for g in order:
            out.loc[g, cyc] = wprev(sub[sub[group_col] == g])
    return out


def plot_trend_overall(adults, lines_txt):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [CYC_YEAR[c] for c in CYC_ORDER]
    overall = [wprev(adults[adults["cycle"] == c]) for c in CYC_ORDER]
    ax.plot(xs, overall, "o-", lw=2.5, color="black", label="Chung")
    lines_txt.append("Tỉ lệ trầm cảm chung (có trọng số) theo chu kỳ:")
    for c, v in zip(CYC_ORDER, overall):
        lines_txt.append(f"  {c}: {v:.1f}%")
    for g, col in [("Female", "#c0504d"), ("Male", "#4f81bd")]:
        ys = [wprev(adults[(adults["cycle"] == c) & (adults["gender"] == g)]) for c in CYC_ORDER]
        ax.plot(xs, ys, "o--", color=col, label=g)
    ax.axvspan(2019, 2020.9, color="grey", alpha=0.12)
    ax.text(2019.9, ax.get_ylim()[0], "  thiếu 2019-20\n  (COVID)", fontsize=7, va="bottom")
    ax.set_xlabel("Năm bắt đầu chu kỳ"); ax.set_ylabel("% trầm cảm (PHQ-9 ≥ 10)")
    ax.set_title("Xu hướng trầm cảm người lớn (18+), NHANES 2007–2023")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "trend_overall.png")


def plot_trend_lines(table, title, fname, cmap_colors):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [CYC_YEAR[c] for c in table.columns]
    for (g, row), col in zip(table.iterrows(), cmap_colors):
        ax.plot(xs, row.values, "o-", label=g, color=col)
    ax.set_xlabel("Năm bắt đầu chu kỳ"); ax.set_ylabel("% trầm cảm (PHQ-9 ≥ 10)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, fname)


def plot_heatmap_age_edu(df):
    mat = pd.DataFrame(index=AGE_ORDER, columns=EDU_ORDER, dtype=float)
    for a in AGE_ORDER:
        for e in EDU_ORDER:
            mat.loc[a, e] = wprev(df[(df["age_group"] == a) & (df["education_label"] == e)])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat.values.astype(float), cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(EDU_ORDER))); ax.set_xticklabels(EDU_ORDER, rotation=20)
    ax.set_yticks(range(len(AGE_ORDER))); ax.set_yticklabels(AGE_ORDER)
    for i in range(len(AGE_ORDER)):
        for j in range(len(EDU_ORDER)):
            v = mat.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 15 else "black")
    ax.set_xlabel("Bậc học"); ax.set_ylabel("Nhóm tuổi")
    ax.set_title("% trầm cảm theo Tuổi × Bậc học (gộp 2007–2023, có trọng số)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, "heatmap_age_education.png")


def plot_delta(df, lines_txt):
    """Thay đổi tỉ lệ 2021-23 so với 2007-08, theo nhóm tuổi & bậc học."""
    first, last = "2007-2008", "2021-2023"
    def delta(col, order):
        a = wprev_by(df[df.cycle == first], col, order)
        b = wprev_by(df[df.cycle == last], col, order)
        return a, b, (b - a)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, order, title) in zip(
        axes, [("age_group", AGE_ORDER, "Theo nhóm tuổi"),
               ("education_label", EDU_ORDER, "Theo bậc học")]):
        a, b, d = delta(col, order)
        x = np.arange(len(order))
        ax.bar(x - 0.2, a.values, 0.4, label=first, color="#9bbb59")
        ax.bar(x + 0.2, b.values, 0.4, label=last, color="#c0504d")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
        ax.set_title(title); ax.set_ylabel("% trầm cảm"); ax.legend()
        lines_txt.append(f"\nThay đổi {first} -> {last} ({title}):")
        for k in order:
            lines_txt.append(f"  {k:<14} {a[k]:.1f}% -> {b[k]:.1f}%  (Δ {d[k]:+.1f})")
    fig.suptitle("Nhóm nào tăng trầm cảm nhanh nhất trong kỷ nguyên smartphone")
    _save(fig, "delta_by_group.png")


def plot_modifiable(adults):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # ngủ
    sl = adults.dropna(subset=["sleep_hours"]).copy()
    sl["g"] = pd.cut(sl["sleep_hours"], [0, 5.9, 6.9, 8.9, 24],
                     labels=["<6h", "6-7h", "7-9h", ">9h"])
    order = ["<6h", "6-7h", "7-9h", ">9h"]
    axes[0, 0].bar(order, wprev_by(sl, "g", order).values, color="#4f81bd")
    axes[0, 0].set_title("Theo thời lượng ngủ"); axes[0, 0].set_ylabel("% trầm cảm")
    # hút thuốc
    sm = {0: "Chưa từng", 1: "Đã bỏ", 2: "Đang hút"}
    vals = wprev_by(adults.assign(s=adults["smoke_status"].map(sm)), "s", list(sm.values()))
    axes[0, 1].bar(vals.index, vals.values, color="#c0504d")
    axes[0, 1].set_title("Theo tình trạng hút thuốc")
    # vận động vừa
    pa = {1.0: "Có", 2.0: "Không"}
    vals = wprev_by(adults.assign(p=adults["phys_moderate"].map(pa)), "p", list(pa.values()))
    axes[1, 0].bar(vals.index, vals.values, color="#9bbb59")
    axes[1, 0].set_title("Có vận động vừa hằng tuần?"); axes[1, 0].set_ylabel("% trầm cảm")
    # ít vận động (sedentary)
    se = adults.dropna(subset=["sedentary_min"]).copy()
    se["g"] = pd.cut(se["sedentary_min"], [0, 240, 480, 1440],
                     labels=["<4h", "4-8h", ">8h"])
    order = ["<4h", "4-8h", ">8h"]
    axes[1, 1].bar(order, wprev_by(se, "g", order).values, color="#8064a2")
    axes[1, 1].set_title("Theo thời gian ngồi/ít vận động mỗi ngày")
    fig.suptitle("Trầm cảm theo yếu tố CAN THIỆP ĐƯỢC (người lớn, có trọng số)")
    _save(fig, "modifiable_factors.png")


def sensitivity_exclude_cycle_l(adults, edu_adults, lines):
    """Kiểm tra độ nhạy phương pháp luận: chu kỳ '2021-2023' (suffix L) là pseudo-cycle CDC
    gộp 3 năm (thay 2019-2020 bị gián đoạn vì COVID), không phải chu kỳ 2 năm chuẩn như các
    chu kỳ khác -> so sánh xu hướng trực tiếp với nó cần thận trọng (xem CYCLES trong
    src/nhanes.py). Ở đây lặp lại 2 kết luận chính, thay '2021-2023' bằng chu kỳ 2 năm chuẩn
    gần nhất (2017-2018), để xem kết luận có đổi không khi bỏ pseudo-cycle."""
    excl, last_normal, first = "2021-2023", "2017-2018", "2007-2008"
    lines.append(f"\n=== SENSITIVITY: loại trừ chu kỳ pseudo {excl} (xem caveat CDC trong src/nhanes.py) ===")

    overall_first = wprev(adults[adults["cycle"] == first])
    overall_last_normal = wprev(adults[adults["cycle"] == last_normal])
    overall_excl = wprev(adults[adults["cycle"] == excl])
    lines.append(
        f"Tỉ lệ chung: {overall_first:.1f}% ({first}) -> {overall_last_normal:.1f}% "
        f"({last_normal}, chu kỳ 2 năm chuẩn gần nhất) -> {overall_excl:.1f}% ({excl}, pseudo-cycle)")

    def top_group(df, col, order, last_cycle):
        a = wprev_by(df[df.cycle == first], col, order)
        b = wprev_by(df[df.cycle == last_cycle], col, order)
        d = b - a
        return d.idxmax(), d[d.idxmax()]

    for col, order, df, label in [
        ("age_group", AGE_ORDER, adults, "nhóm tuổi"),
        ("education_label", EDU_ORDER, edu_adults, "bậc học"),
    ]:
        g_excl, d_excl = top_group(df, col, order, excl)
        g_normal, d_normal = top_group(df, col, order, last_normal)
        consistent = g_excl == g_normal
        lines.append(
            f"  Nhóm tăng nhanh nhất theo {label}: dùng {excl} -> '{g_excl}' (+{d_excl:.1f}); "
            f"dùng {last_normal} (loại pseudo-cycle) -> '{g_normal}' (+{d_normal:.1f}) "
            f"— {'NHẤT QUÁN' if consistent else 'ĐỔI KẾT LUẬN, cần đọc thận trọng'}")


def main():
    df = nhanes.load_pooled()
    adults = df[(df["age"] >= 18) & df["dep_risk"].notna()].copy()
    edu_adults = df[(df["age"] >= 20) & df["dep_risk"].notna() &
                    df["education_label"].notna()].copy()
    print(f"Phân tích {len(adults):,} người lớn có PHQ-9 hợp lệ")

    lines = []
    plot_trend_overall(adults, lines)
    plot_trend_lines(trend_table(adults, "age_group", AGE_ORDER),
                     "Trầm cảm theo nhóm tuổi qua thời gian", "trend_by_age.png",
                     ["#2c7fb8", "#41b6c4", "#a1dab4", "#fec44f", "#d95f0e"])
    plot_trend_lines(trend_table(edu_adults, "education_label", EDU_ORDER),
                     "Trầm cảm theo bậc học qua thời gian (20+)", "trend_by_education.png",
                     ["#d7301f", "#fc8d59", "#fdcc8a", "#74a9cf", "#0570b0"])
    plot_heatmap_age_edu(edu_adults)
    plot_delta(edu_adults, lines)
    plot_modifiable(adults)
    sensitivity_exclude_cycle_l(adults, edu_adults, lines)

    (C.METRIC_DIR / "nhanes_summary.txt").write_text("\n".join(lines))
    print(f"\nSố liệu chính -> outputs/metrics/nhanes_summary.txt")


if __name__ == "__main__":
    main()
