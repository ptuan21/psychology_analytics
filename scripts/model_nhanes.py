"""
ĐI SÂU: hồi quy logistic ĐA BIẾN có trọng số (NHANES, người lớn 20+) để định lượng
đóng góp RIÊNG của từng yếu tố tới nguy cơ trầm cảm (PHQ-9 ≥ 10), khi đã kiểm soát
các yếu tố còn lại. Đầu ra là Odds Ratio (OR) + khoảng tin cậy 95%.

  OR > 1: làm TĂNG nguy cơ;  OR < 1: yếu tố BẢO VỆ.

Hai mô hình:
  (1) Mô hình đầy đủ      -> OR cho mọi yếu tố (forest plot + bảng).
  (2) Mô hình tương tác   -> kiểm định "người trẻ tăng nhanh hơn" (age_group × year).

Phương pháp: trọng số khảo sát (chuẩn hoá về cỡ mẫu) cho ước lượng OR, và
KHOẢNG TIN CẬY DESIGN-BASED — sai số chuẩn cụm-vững theo PSU lồng trong tầng
(SDMVPSU trong SDMVSTRA), phản ánh thiết kế chọn mẫu phức tạp của NHANES.
CI vì thế rộng & trung thực hơn so với SE ngây thơ từ freq_weights.

Chạy:  python3 scripts/model_nhanes.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as CFG
from src import nhanes

FIG = CFG.FIG_DIR / "nhanes"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các cột đã gán nhãn + chuẩn hoá, lọc complete-case cho mô hình."""
    d = df[(df["age"] >= 20) & df["dep_risk"].notna()].copy()
    d["dep"] = d["dep_risk"].astype(int)

    # Categorical với HẠNG MỤC THAM CHIẾU đặt ĐẦU TIÊN (patsy lấy level đầu làm chuẩn)
    def cat(series, order):
        return pd.Categorical(series, categories=order)
    d["age_group"] = cat(d["age_group"], ["36-50", "18-25", "26-35", "51-65", "66+"])
    d["gender"] = cat(d["gender"], ["Male", "Female"])
    d["edu"] = cat(d["education_label"],
                   ["College grad", "<9th", "9-11th", "HS/GED", "Some college"])
    d["race_l"] = cat(d["race"].map(nhanes.RACE_LABELS),
                      ["White", "Mexican-Am", "Other Hispanic", "Black", "Other/Multi"])
    d["marital"] = cat(d["marital"], ["partnered", "previously", "never"])
    d["smoke_l"] = cat(d["smoke_status"].map(nhanes.SMOKE_LABELS),
                       ["Chưa từng", "Đã bỏ", "Đang hút"])
    sl = pd.cut(d["sleep_hours"], [0, 5.9, 6.9, 8.9, 24], labels=["<6h", "6-7h", "7-9h", ">9h"])
    d["sleep_g"] = cat(sl, ["7-9h", "<6h", "6-7h", ">9h"])
    # chuẩn hoá biến liên tục -> OR theo 1 độ lệch chuẩn
    for c in ["income_poverty", "sedentary_min"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    d["genhealth"] = d["gen_health"]          # 1..5, OR cho mỗi mức xấu hơn
    d["year_c"] = (d["year"] - 2007) / 10.0   # mỗi 10 năm
    # trọng số chuẩn hoá về cỡ mẫu
    cols = ["dep", "age_group", "gender", "edu", "race_l", "marital", "smoke_l",
            "sleep_g", "income_poverty_z", "sedentary_min_z", "genhealth",
            "year_c", "weight", "strata", "psu"]
    d = d.dropna(subset=cols)
    d["w"] = d["weight"] * len(d) / d["weight"].sum()
    # cụm = PSU lồng trong tầng -> dùng cho SE design-based (cụm-vững)
    d["cluster"] = d["strata"].astype(int) * 100 + d["psu"].astype(int)
    return d


def fit_full(d):
    f = ("dep ~ C(age_group) + C(gender) + C(edu) + C(race_l) + C(marital)"
         " + C(smoke_l) + C(sleep_g) + income_poverty_z + sedentary_min_z"
         " + genhealth + year_c")
    glm = smf.glm(f, data=d, family=sm.families.Binomial(), freq_weights=d["w"])
    # SE design-based: cụm-vững theo PSU-trong-tầng (đúng thiết kế NHANES, CI không còn xấp xỉ hẹp)
    return glm.fit(cov_type="cluster", cov_kwds={"groups": d["cluster"].values})


PRETTY = {  # rút gọn tên term cho dễ đọc
    "C(age_group)[T.18-25]": "Tuổi 18-25 (vs 36-50)",
    "C(age_group)[T.26-35]": "Tuổi 26-35",
    "C(age_group)[T.51-65]": "Tuổi 51-65",
    "C(age_group)[T.66+]": "Tuổi 66+",
    "C(gender)[T.Female]": "Nữ (vs Nam)",
    "C(edu)[T.<9th]": "Học vấn <lớp 9",
    "C(edu)[T.9-11th]": "Học vấn lớp 9-11",
    "C(edu)[T.HS/GED]": "Tốt nghiệp PT",
    "C(edu)[T.Some college]": "Học CĐ dở dang",
    "C(race_l)[T.Black]": "Da đen (vs Trắng)",
    "C(race_l)[T.Mexican-Am]": "Mexican-Am",
    "C(race_l)[T.Other Hispanic]": "Hispanic khác",
    "C(race_l)[T.Other/Multi]": "Chủng tộc khác",
    "C(marital)[T.never]": "Chưa kết hôn (vs có bạn đời)",
    "C(marital)[T.previously]": "Ly hôn/goá",
    "C(smoke_l)[T.Đang hút]": "Đang hút thuốc",
    "C(smoke_l)[T.Đã bỏ]": "Đã bỏ thuốc",
    "C(sleep_g)[T.<6h]": "Ngủ <6h (vs 7-9h)",
    "C(sleep_g)[T.6-7h]": "Ngủ 6-7h",
    "C(sleep_g)[T.>9h]": "Ngủ >9h",
    "income_poverty_z": "Thu nhập (mỗi +1 SD)",
    "sedentary_min_z": "Ngồi nhiều (mỗi +1 SD)",
    "genhealth": "Sức khoẻ kém hơn (mỗi +1 mức)",
    "year_c": "Mỗi 10 năm trôi qua",
}


def odds_table(res) -> pd.DataFrame:
    p = res.params.drop("Intercept")
    ci = res.conf_int().drop("Intercept")
    t = pd.DataFrame({
        "term": [PRETTY.get(i, i) for i in p.index],
        "OR": np.exp(p.values),
        "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values),
        "p": res.pvalues.drop("Intercept").values,
    })
    return t.sort_values("OR", ascending=False).reset_index(drop=True)


def forest_plot(t, path):
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 10))
    y = range(len(t))
    colors = ["#c0504d" if o > 1 else "#4f81bd" for o in t["OR"]]
    ax.errorbar(t["OR"], y, xerr=[t["OR"] - t["lo"], t["hi"] - t["OR"]],
                fmt="none", ecolor="grey", elinewidth=1, capsize=2, zorder=1)
    ax.scatter(t["OR"], y, color=colors, zorder=2, s=30)
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.set_yticks(list(y)); ax.set_yticklabels(t["term"], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Odds Ratio (thang log) — >1 tăng nguy cơ, <1 bảo vệ")
    ax.set_title("Yếu tố độc lập của trầm cảm (logistic đa biến, có trọng số)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def interaction_test(d):
    """Kiểm định người trẻ tăng nhanh hơn: age_group × year_c."""
    f = "dep ~ C(age_group) * year_c"
    res = smf.glm(f, data=d, family=sm.families.Binomial(), freq_weights=d["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["cluster"].values})
    rows = []
    for term in res.params.index:
        if ":year_c" in term:
            lab = PRETTY.get(term.split(":")[0], term.split(":")[0]).split(" (")[0]
            rows.append((lab, np.exp(res.params[term]), res.pvalues[term]))
    return rows


def main():
    d = prepare(nhanes.load_pooled())
    print(f"Mô hình trên {len(d):,} người lớn (complete-case, 20+)\n")

    res = fit_full(d)
    t = odds_table(res)
    t.round(3).to_csv(CFG.METRIC_DIR / "nhanes_logit_or.csv", index=False)
    forest_plot(t, FIG / "odds_ratios.png")
    print("  + outputs/figures/nhanes/odds_ratios.png")
    print("  + outputs/metrics/nhanes_logit_or.csv\n")

    print("YẾU TỐ NGUY CƠ MẠNH NHẤT (OR cao nhất):")
    for _, r in t.head(6).iterrows():
        print(f"  {r['term']:<32} OR={r['OR']:.2f}  [{r['lo']:.2f}-{r['hi']:.2f}]  p={r['p']:.1e}")
    print("\nYẾU TỐ BẢO VỆ MẠNH NHẤT (OR thấp nhất):")
    for _, r in t.tail(4).iloc[::-1].iterrows():
        print(f"  {r['term']:<32} OR={r['OR']:.2f}  [{r['lo']:.2f}-{r['hi']:.2f}]  p={r['p']:.1e}")

    print("\nKIỂM ĐỊNH TƯƠNG TÁC tuổi × thời gian (OR mỗi 10 năm, so với nhóm 36-50):")
    for lab, orr, pv in interaction_test(d):
        flag = " *** nhanh hơn rõ rệt" if (orr > 1.05 and pv < 0.05) else ""
        print(f"  {lab:<14} OR_tương_tác={orr:.2f}  p={pv:.1e}{flag}")


if __name__ == "__main__":
    main()
