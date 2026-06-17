"""
Cấu hình tập trung cho toàn bộ pipeline: đường dẫn, nhóm cột, ngưỡng severity.
Sửa mọi hằng số ở đây — các module khác chỉ import, không hard-code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "data.csv"          # file gốc, ngăn cách bằng TAB
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
METRIC_DIR = OUT_DIR / "metrics"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Nhóm câu hỏi DASS-42  (mỗi subscale 14 câu)
# Nguồn: Lovibond & Lovibond (1995)
# ---------------------------------------------------------------------------
DASS_SUBSCALES = {
    "depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    "anxiety":    [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    "stress":     [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39],
}
# Cột câu trả lời tương ứng (Qn A). Thang gốc trong file là 1..4 -> trừ 1 để về 0..3.
DASS_ANSWER_COLS = [f"Q{i}A" for i in range(1, 43)]

# ---------------------------------------------------------------------------
# Ngưỡng severity DASS-42 (điểm = TỔNG 14 câu, mỗi câu 0..3, range 0..42)
# (label, lower, upper)  — upper là cận trên đã bao gồm
# ---------------------------------------------------------------------------
SEVERITY_CUTOFFS = {
    "depression": [
        ("Normal", 0, 9), ("Mild", 10, 13), ("Moderate", 14, 20),
        ("Severe", 21, 27), ("Extremely Severe", 28, 42),
    ],
    "anxiety": [
        ("Normal", 0, 7), ("Mild", 8, 9), ("Moderate", 10, 14),
        ("Severe", 15, 19), ("Extremely Severe", 20, 42),
    ],
    "stress": [
        ("Normal", 0, 14), ("Mild", 15, 18), ("Moderate", 19, 25),
        ("Severe", 26, 33), ("Extremely Severe", 34, 42),
    ],
}
SEVERITY_ORDER = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

# Nếu True: gộp "Severe" + "Extremely Severe" -> "Severe" (về đúng 4 lớp như yêu cầu)
COLLAPSE_SEVERE = False

# ---------------------------------------------------------------------------
# Chế độ bài toán  ( đổi 1 dòng -> đổi toàn pipeline )
#   "multiclass"  : phân loại 5 mức severity (Normal..Extremely Severe)
#   "binary"      : phân loại nhị phân "nguy cơ cao" (Moderate trở lên)  <- AUC ~0.81
#   "regression"  : hồi quy điểm DASS liên tục 0..42
# ---------------------------------------------------------------------------
TASK_MODE = "binary"

# Ngưỡng "nguy cơ cao" = cận dưới của mức Moderate, suy ra từ SEVERITY_CUTOFFS.
HIGH_RISK_CUTOFF = {
    name: next(lo for lab, lo, hi in cuts if lab == "Moderate")
    for name, cuts in SEVERITY_CUTOFFS.items()
}  # -> {'depression': 14, 'anxiety': 10, 'stress': 19}
BINARY_LABELS = ["Không/Nhẹ", "Nguy cơ cao"]  # 0, 1

# ---------------------------------------------------------------------------
# TIPI (Big Five rút gọn, thang 1..7). 0 = thiếu.
# Quy tắc tính điểm Gosling et al. (2003): mỗi chiều = trung bình của
# (item thuận, item đảo). Item đảo được reverse: 8 - score.
# ---------------------------------------------------------------------------
TIPI_COLS = [f"TIPI{i}" for i in range(1, 11)]
BIG_FIVE = {
    # tên_chiều: (item_thuận, item_đảo)
    "extraversion":        (1, 6),
    "agreeableness":       (7, 2),
    "conscientiousness":   (3, 8),
    "emotional_stability": (9, 4),
    "openness":            (5, 10),
}

# ---------------------------------------------------------------------------
# Cột kiểm tra tính hợp lệ
# ---------------------------------------------------------------------------
VCL_FAKE_WORDS = ["VCL6", "VCL9", "VCL12"]   # từ giả -> tích = trả lời thiếu trung thực
MAX_FAKE_WORDS_ALLOWED = 0                    # loại nếu tích > ngưỡng này
AGE_MIN, AGE_MAX = 13, 100                    # khoảng tuổi hợp lý

# ---------------------------------------------------------------------------
# Đặc trưng cho MÔ HÌNH (chỉ nhân khẩu học + tính cách — KHÔNG dùng câu DASS)
# ---------------------------------------------------------------------------
# Đặc trưng số
NUMERIC_FEATURES = ["age", "familysize"] + [f"big5_{k}" for k in BIG_FIVE]
# Đặc trưng phân loại (0 = thiếu -> xử lý như một hạng mục riêng)
CATEGORICAL_FEATURES = [
    "education", "urban", "gender", "engnat", "religion",
    "orientation", "race", "voted", "married", "hand", "screensize",
]
