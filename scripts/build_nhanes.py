"""
Dựng bảng NHANES gộp 2007–2023 từ các file .xpt thật (data/nhanes/raw/)
và kiểm tra nhanh xu hướng trầm cảm theo thời gian.

Chạy:  python3 scripts/build_nhanes.py
Đầu ra: data/nhanes/nhanes_pooled.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nhanes


def main():
    df = nhanes.build_pooled()

    # chỉ người lớn 18+ có PHQ-9 hợp lệ, dùng để kiểm tra tỉ lệ thô
    adults = df[(df["age"] >= 18) & df["dep_risk"].notna()]
    print("\nTỉ lệ trầm cảm (PHQ-9>=10) theo chu kỳ — người lớn 18+ (thô, chưa trọng số):")
    tab = adults.groupby("cycle")["dep_risk"].agg(["mean", "count"])
    for cyc, row in tab.iterrows():
        print(f"  {cyc}: {row['mean']*100:5.1f}%  (n={int(row['count']):,})")

    print("\nĐộ phủ biến (tỉ lệ không thiếu):")
    for c in ["phq9_total", "education", "sleep_hours", "alc_drinks_day",
              "phys_moderate", "sedentary_min", "smoke_status", "gen_health"]:
        print(f"  {c:<16} {df[c].notna().mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
