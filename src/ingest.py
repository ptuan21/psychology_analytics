"""
Nhập & kiểm tra dữ liệu THU THẬP MỚI (có các cột mở rộng ở codebook_extension.txt).

Chức năng:
  - validate(): kiểm tra đủ cột, đúng khoảng giá trị; trả về danh sách cảnh báo.
  - add_derived(): tính điểm dẫn xuất (UCLA-3, AUDIT-C, SDS) + cờ an toàn.
  - load_collected(): đọc file TAB -> validate -> add_derived.
  - write_template(): xuất CSV mẫu (header gồm cột gốc + cột mở rộng) để đi thu thập.

Lưu ý an toàn: safety_flag=1 nghĩa là người trả lời có dấu hiệu nguy cơ tự hại
-> hệ thống khảo sát PHẢI hiển thị nguồn hỗ trợ khủng hoảng ngay (xem codebook).
"""
import numpy as np
import pandas as pd

from . import config as C


def validate(df: pd.DataFrame) -> list[str]:
    """Trả về danh sách cảnh báo (rỗng = hợp lệ)."""
    warns = []
    missing = [c for c in C.EXT_COLUMNS if c not in df.columns]
    if missing:
        warns.append(f"Thiếu {len(missing)} cột mở rộng: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    for col, (lo, hi) in C.EXT_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = ((s < lo) if lo is not None else False) | ((s > hi) if hi is not None else False)
        n = int(np.nansum(np.asarray(bad, dtype=float)))
        if n:
            warns.append(f"{col}: {n} giá trị ngoài khoảng [{lo},{hi}]")
    return warns


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột điểm dẫn xuất + safety_flag (bỏ qua nếu thiếu cột nguồn)."""
    df = df.copy()
    if set(["ucla_companion", "ucla_leftout", "ucla_isolated"]).issubset(df.columns):
        df["ucla_total"] = df[["ucla_companion", "ucla_leftout", "ucla_isolated"]].sum(axis=1)
    if set(["auditc_freq", "auditc_amount", "auditc_binge"]).issubset(df.columns):
        df["auditc_total"] = df[["auditc_freq", "auditc_amount", "auditc_binge"]].sum(axis=1)
    if set(C.EXT_FUNCTION[:3]).issubset(df.columns):
        df["sds_total"] = df[C.EXT_FUNCTION[:3]].sum(axis=1)
    safety_cols = [c for c in C.EXT_SAFETY if c in df.columns]
    if safety_cols:
        df["safety_flag"] = (df[safety_cols] == 1).any(axis=1).astype(int)
    return df


def load_collected(path: str, verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    warns = validate(df)
    if verbose:
        if warns:
            print("  [ingest] CẢNH BÁO:")
            for w in warns:
                print(f"    - {w}")
        else:
            print("  [ingest] hợp lệ, không có cảnh báo")
    df = add_derived(df)
    if "safety_flag" in df.columns and verbose:
        n = int(df["safety_flag"].sum())
        print(f"  [ingest] safety_flag dương: {n} người -> cần hiển thị nguồn hỗ trợ khủng hoảng")
    return df


def write_template(path: str = None) -> str:
    """Tạo CSV mẫu: header = cột gốc (từ data.csv) + cột mở rộng, 0 dòng dữ liệu."""
    path = path or str(C.DATA_DIR / "survey_template.csv")
    base = list(pd.read_csv(C.DATA_FILE, sep="\t", nrows=0).columns)
    header = base + [c for c in C.EXT_COLUMNS if c not in base]
    pd.DataFrame(columns=header).to_csv(path, sep="\t", index=False)
    return path
