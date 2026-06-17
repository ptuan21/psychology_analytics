"""
Hài hoà dữ liệu NHANES (CDC) nhiều chu kỳ 2007–2023 thành một bảng sạch.

Vì sao cần module riêng: tên biến NHANES đổi giữa các chu kỳ (vd giờ ngủ
SLD010H -> SLD012; rượu ALQ120Q -> ALQ121). Ở đây map từng biến theo chu kỳ
rồi đổi về tên chuẩn, tính PHQ-9 -> mức độ trầm cảm, gộp lại kèm cột `cycle`/`year`.

Dữ liệu là THẬT, tải từ wwwn.cdc.gov (xem scripts/fetch_nhanes hoặc raw/*.xpt).
PHQ-9: 9 mục DPQ010..DPQ090, mỗi mục 0–3 (7=từ chối, 9=không biết -> thiếu).
Tổng 0–27. Ngưỡng chuẩn: >=10 = trầm cảm có ý nghĩa lâm sàng.
Lưu ý: học vấn (DMDEDUC2) chỉ hỏi người >=20 tuổi -> phân tích theo bậc học là ở người lớn.
"""
import warnings

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")

RAW = C.DATA_DIR / "nhanes" / "raw"
POOLED = C.DATA_DIR / "nhanes" / "nhanes_pooled.csv"

# suffix -> (năm bắt đầu, nhãn chu kỳ)
CYCLES = {
    "E": (2007, "2007-2008"), "F": (2009, "2009-2010"), "G": (2011, "2011-2012"),
    "H": (2013, "2013-2014"), "I": (2015, "2015-2016"), "J": (2017, "2017-2018"),
    "L": (2021, "2021-2023"),
}
PHQ_ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]  # DPQ010..DPQ090 (9 mục được chấm)

EDU_LABELS = {1: "<9th", 2: "9-11th", 3: "HS/GED", 4: "Some college", 5: "College grad"}
AGE_BINS = [12, 18, 26, 36, 51, 66, 200]
AGE_LABELS = ["12-17", "18-25", "26-35", "36-50", "51-65", "66+"]
DEP_CUTS = [(0, 4, "None-minimal"), (5, 9, "Mild"), (10, 14, "Moderate"),
            (15, 19, "Mod-severe"), (20, 27, "Severe")]


def _read(comp, suf):
    f = RAW / f"{comp}_{suf}.xpt"
    return pd.read_sas(f, format="xport") if f.exists() else None


def _na(s, codes):
    if not isinstance(s, pd.Series):  # cột vắng ở chu kỳ này -> scalar NaN
        return np.nan
    return s.replace(list(codes), np.nan)


def _smoke(demo_merged):
    """0=chưa từng, 1=đã bỏ, 2=đang hút (từ SMQ020/SMQ040)."""
    q20 = _na(demo_merged.get("SMQ020", np.nan), [7, 9])
    q40 = _na(demo_merged.get("SMQ040", np.nan), [7, 9])
    out = pd.Series(np.nan, index=demo_merged.index)
    out[q20 == 2] = 0
    out[(q20 == 1) & (q40 == 3)] = 1
    out[(q40 == 1) | (q40 == 2)] = 2
    return out


def _marital(demo, suf):
    if suf == "L":  # DMDMARTZ: 1=married/partner,2=prev,3=never
        m = _na(demo.get("DMDMARTZ", np.nan), [77, 99])
        return m.map({1: "partnered", 2: "previously", 3: "never"})
    m = _na(demo.get("DMDMARTL", np.nan), [77, 99])  # 1..6
    return m.map({1: "partnered", 6: "partnered", 2: "previously",
                  3: "previously", 4: "previously", 5: "never"})


def build_cycle(suf):
    """Trả về DataFrame đã hài hoá cho một chu kỳ, hoặc None nếu thiếu lõi."""
    demo, dpq = _read("DEMO", suf), _read("DPQ", suf)
    if demo is None or dpq is None:
        return None
    start, label = CYCLES[suf]

    # gộp các thành phần vào DEMO theo SEQN
    df = demo.copy()
    for comp in ["DPQ", "SLQ", "ALQ", "PAQ", "SMQ", "HUQ"]:
        part = _read(comp, suf)
        if part is not None:
            df = df.merge(part, on="SEQN", how="left", suffixes=("", f"_{comp}"))

    out = pd.DataFrame({"SEQN": df["SEQN"].astype("int64")})
    out["cycle"] = label
    out["year"] = start

    # nhân khẩu
    out["age"] = df["RIDAGEYR"]
    out["age_group"] = pd.cut(df["RIDAGEYR"], AGE_BINS, labels=AGE_LABELS, right=False)
    out["gender"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    edu = _na(df["DMDEDUC2"], [7, 9])
    out["education"] = edu
    out["education_label"] = edu.map(EDU_LABELS)
    out["race"] = df["RIDRETH1"]
    out["income_poverty"] = df.get("INDFMPIR", np.nan)
    out["marital"] = _marital(df, suf)

    # trọng số khảo sát (cho ước lượng tỷ lệ đúng dân số)
    out["weight"] = df.get("WTMEC2YR", np.nan)
    out["psu"] = df.get("SDMVPSU", np.nan)
    out["strata"] = df.get("SDMVSTRA", np.nan)

    # PHQ-9 -> tổng, mức độ, nhị phân
    phq = df[PHQ_ITEMS].round().replace([7, 9], np.nan)  # round: khử nhiễu float XPORT
    total = phq.sum(axis=1)
    total[phq.isna().any(axis=1)] = np.nan  # cần đủ 9 mục
    out["phq9_total"] = total
    out["dep_level"] = total.apply(
        lambda t: next((lab for lo, hi, lab in DEP_CUTS if lo <= t <= hi), np.nan)
        if pd.notna(t) else np.nan)
    out["dep_risk"] = (total >= 10).where(total.notna())

    # yếu tố can thiệp được (hài hoà tên biến theo chu kỳ)
    sleep = df["SLD012"] if "SLD012" in df else df.get("SLD010H", np.nan)
    out["sleep_hours"] = _na(sleep, [77, 99]).clip(upper=14)
    out["alc_drinks_day"] = _na(df.get("ALQ130", np.nan), [777, 999])
    out["phys_moderate"] = _na(df.get("PAQ665", np.nan), [7, 9])  # 1=có,2=không (thiếu ở 2021-2023)
    out["sedentary_min"] = _na(df.get("PAD680", np.nan), [7777, 9999])
    out["smoke_status"] = _smoke(df)
    out["gen_health"] = _na(df.get("HUQ010", np.nan), [7, 9])  # 1=Excellent..5=Poor
    return out


def build_pooled(save=True, verbose=True) -> pd.DataFrame:
    frames = [build_cycle(s) for s in CYCLES]
    pooled = pd.concat([f for f in frames if f is not None], ignore_index=True)
    if save:
        POOLED.parent.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(POOLED, index=False)
    if verbose:
        print(f"Gộp {len(pooled):,} người, {pooled['cycle'].nunique()} chu kỳ -> {POOLED.name}")
    return pooled


def load_pooled() -> pd.DataFrame:
    return pd.read_csv(POOLED)


# ---------------------------------------------------------------------------
# Tiện ích ước lượng CÓ TRỌNG SỐ (dùng chung cho các script phân tích)
# ---------------------------------------------------------------------------
def wprev(d: pd.DataFrame, col: str = "dep_risk") -> float:
    """Tỉ lệ dương của `col` (%) có trọng số WTMEC2YR. NaN nếu rỗng."""
    d = d.dropna(subset=[col, "weight"])
    if len(d) == 0 or d["weight"].sum() == 0:
        return np.nan
    return 100 * np.average(d[col].astype(float), weights=d["weight"])


def wprev_by(df: pd.DataFrame, col: str, order, target: str = "dep_risk") -> pd.Series:
    """Tỉ lệ có trọng số theo từng giá trị của `col`, theo thứ tự `order`."""
    return pd.Series({k: wprev(df[df[col] == k], target) for k in order})


# Nhãn hiển thị dùng chung
RACE_LABELS = {1: "Mexican-Am", 2: "Other Hispanic", 3: "White", 4: "Black", 5: "Other/Multi"}
GENHEALTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
SMOKE_LABELS = {0: "Chưa từng", 1: "Đã bỏ", 2: "Đang hút"}
INCOME_BINS = [0, 1, 2, 4, 5.01]
INCOME_LABELS = ["Nghèo (<1)", "Cận nghèo 1-2", "Trung bình 2-4", "Khá ≥4"]
