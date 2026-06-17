"""
Tính điểm:
  - 3 thang DASS (depression / anxiety / stress) + gán nhãn severity.
  - 5 chiều Big Five từ TIPI (dùng làm đặc trưng cho mô hình).

Lưu ý quan trọng: nhãn severity (target) được tính TỪ câu DASS, còn đặc trưng
mô hình thì KHÔNG dùng câu DASS -> tránh rò rỉ nhãn (label leakage).
"""
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# DASS -> điểm subscale + severity
# ---------------------------------------------------------------------------
def dass_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột <subscale>_score (0..42) cho mỗi thang."""
    df = df.copy()
    for name, items in C.DASS_SUBSCALES.items():
        cols = [f"Q{i}A" for i in items]
        # thang gốc 1..4 -> trừ 1 về 0..3 rồi cộng 14 câu
        df[f"{name}_score"] = (df[cols] - 1).sum(axis=1)
    return df


def _to_severity(score: int, cutoffs) -> str:
    for label, lo, hi in cutoffs:
        if lo <= score <= hi:
            return label
    return cutoffs[-1][0]


def dass_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột <subscale>_level (nhãn severity) cho mỗi thang."""
    df = df.copy()
    for name, cutoffs in C.SEVERITY_CUTOFFS.items():
        lvl = df[f"{name}_score"].apply(lambda s: _to_severity(s, cutoffs))
        if C.COLLAPSE_SEVERE:
            lvl = lvl.replace("Extremely Severe", "Severe")
        df[f"{name}_level"] = lvl
    return df


# ---------------------------------------------------------------------------
# TIPI -> Big Five
# ---------------------------------------------------------------------------
def big_five(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm 5 cột big5_<chiều>. Item đảo được reverse (8 - score)."""
    df = df.copy()
    for dim, (plain, reverse) in C.BIG_FIVE.items():
        a = df[f"TIPI{plain}"]
        b = 8 - df[f"TIPI{reverse}"]          # reverse-scored
        df[f"big5_{dim}"] = (a + b) / 2.0     # NaN nếu TIPI thiếu -> impute ở pipeline
    return df


def build_targets_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tiện ích: chạy cả 3 bước scoring trên một DataFrame đã làm sạch."""
    return big_five(dass_severity(dass_scores(df)))


# ---------------------------------------------------------------------------
# Dựng biến mục tiêu theo TASK_MODE
# ---------------------------------------------------------------------------
def make_target(df: pd.DataFrame, target: str, mode: str):
    """
    Trả về (y, labels, kind):
      - mode "multiclass": y = mã 0..k của severity; labels = tên mức; kind="clf"
      - mode "binary":     y = 0/1 nguy cơ cao;      labels = BINARY_LABELS; kind="clf"
      - mode "regression": y = điểm 0..42 (float);   labels = None;          kind="reg"
    """
    if mode == "multiclass":
        y_str = df[f"{target}_level"]
        order = C.SEVERITY_ORDER[:] if not C.COLLAPSE_SEVERE else \
            [l for l in C.SEVERITY_ORDER if l != "Extremely Severe"]
        labels = [l for l in order if l in set(y_str)]
        code = {lab: i for i, lab in enumerate(labels)}
        return y_str.map(code).astype(int), labels, "clf"

    if mode == "binary":
        y = (df[f"{target}_score"] >= C.HIGH_RISK_CUTOFF[target]).astype(int)
        return y, C.BINARY_LABELS, "clf"

    if mode == "regression":
        return df[f"{target}_score"].astype(float), None, "reg"

    raise ValueError(f"TASK_MODE không hợp lệ: {mode!r}")
