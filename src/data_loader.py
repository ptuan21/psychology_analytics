"""
Tải dữ liệu thô và làm sạch / lọc tính hợp lệ.

Các bước làm sạch (đều bật/tắt được qua tham số):
  1. Loại người tích vào "từ giả" trong VCL (trả lời thiếu trung thực).
  2. Loại tuổi phi lý (ngoài [AGE_MIN, AGE_MAX]).
  3. Đưa các giá trị 0 = "thiếu" về NaN (TIPI và một số cột nhân khẩu).
"""
import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    """Đọc file gốc (TAB-separated)."""
    return pd.read_csv(C.DATA_FILE, sep="\t")


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Trả về DataFrame đã làm sạch (không sửa df gốc)."""
    df = df.copy()
    n0 = len(df)
    log = (lambda m: print(f"  [clean] {m}")) if verbose else (lambda m: None)

    # 1) Lọc tính hợp lệ bằng từ giả VCL
    fake_checked = df[C.VCL_FAKE_WORDS].sum(axis=1)
    keep = fake_checked <= C.MAX_FAKE_WORDS_ALLOWED
    log(f"loại {(~keep).sum():>6} dòng tích >{C.MAX_FAKE_WORDS_ALLOWED} từ giả (validity)")
    df = df[keep]

    # 2) Lọc tuổi hợp lý
    keep = df["age"].between(C.AGE_MIN, C.AGE_MAX)
    log(f"loại {(~keep).sum():>6} dòng tuổi ngoài [{C.AGE_MIN},{C.AGE_MAX}]")
    df = df[keep]

    # 3) 0 = thiếu -> NaN cho TIPI (thang hợp lệ là 1..7)
    df[C.TIPI_COLS] = df[C.TIPI_COLS].replace(0, np.nan)

    # 4) familysize bất thường (0 = thiếu, cap ở 20 cho ngoại lai)
    df["familysize"] = df["familysize"].replace(0, np.nan).clip(upper=20)

    log(f"còn lại {len(df):>6}/{n0} dòng ({len(df)/n0:.1%})")
    return df.reset_index(drop=True)


def load_clean(verbose: bool = True) -> pd.DataFrame:
    return clean(load_raw(), verbose=verbose)
