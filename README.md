# Dự đoán mức độ Trầm cảm / Lo âu / Stress (DASS) từ nhân khẩu học + tính cách

Bài toán **phân loại đa lớp**: từ điểm DASS-42 tính ra nhãn severity
(*Normal / Mild / Moderate / Severe / Extremely Severe*) cho 3 thang
**depression, anxiety, stress**, rồi huấn luyện mô hình dự đoán nhãn đó
**chỉ bằng nhân khẩu học + tính cách (TIPI/Big Five)** — KHÔNG dùng câu trả lời DASS.

> Mục tiêu: ước lượng nhóm rủi ro của một người **mà không cần** họ làm bài DASS,
> chỉ dựa trên thông tin nền và một bài tính cách 10 câu.

## Cấu trúc thư mục

```
psy_analysis/
├── data/                      # dữ liệu gốc (data.csv TAB-separated, codebook.txt)
├── src/                       # mã nguồn module hoá
│   ├── config.py              # MỌI hằng số: nhóm cột, ngưỡng severity, đường dẫn
│   ├── data_loader.py         # tải + làm sạch + lọc tính hợp lệ (từ giả VCL, tuổi)
│   ├── scoring.py             # điểm DASS -> severity; TIPI -> Big Five
│   ├── features.py            # chọn đặc trưng + pipeline tiền xử lý (impute/scale/onehot)
│   ├── model.py               # định nghĩa model + xử lý mất cân bằng lớp
│   └── evaluate.py            # metric, confusion matrix, feature importance
├── scripts/
│   ├── run_pipeline.py        # train + đánh giá end-to-end (theo TASK_MODE)
│   ├── run_eda.py             # sinh biểu đồ trực quan hoá dữ liệu
│   └── analyze_framing.py     # phân tích vì sao macro-F1 thấp (3 cách đóng khung)
├── outputs/                   # figures + metrics có trên git; models/ thì không
│   ├── figures/               # cm_*, scatter_*, importance_*  +  figures/eda/*
│   ├── metrics/               # metrics_*.csv, report_*.txt, summary_*.json
│   └── models/                # model_*.joblib (BỊ .gitignore — tự sinh lại)
├── requirements.txt
└── README.md
```

## Cách chạy

```bash
pip install -r requirements.txt
python3 scripts/run_eda.py          # biểu đồ EDA -> outputs/figures/eda/
python3 scripts/run_pipeline.py     # train + đánh giá -> outputs/
```

## Ba chế độ bài toán — đổi 1 dòng trong `src/config.py`

```python
TASK_MODE = "binary"   # "multiclass" | "binary" | "regression"
```

| Chế độ | Bài toán | Chọn model theo | Kết quả |
|--------|----------|-----------------|---------|
| `multiclass` | 5 mức severity | macro-F1 | ~0.35 (xem mục "Đọc kết quả") |
| **`binary`** | **nguy cơ cao (Moderate+)** | **ROC-AUC** | **AUC 0.80–0.83** ⭐ khuyến nghị |
| `regression` | điểm liên tục 0–42 | R² | R² 0.33–0.42, MAE ~6–8 điểm |

## Biểu đồ EDA (`outputs/figures/eda/`)

| File | Nội dung |
|------|----------|
| `eda_severity_distribution.png` | phân bố 5 mức severity theo 3 thang |
| `eda_score_histograms.png` | phân bố điểm 0–42 + ngưỡng nguy cơ |
| `eda_correlation_heatmap.png` | tương quan điểm DASS ↔ Big Five |
| `eda_bigfive_vs_dass.png` | Big Five dự báo mức độ nặng thế nào |
| `eda_demographics.png` | tuổi, giới, học vấn, ngôn ngữ |
| `eda_score_by_group.png` | điểm trung bình theo giới & nhóm tuổi |

## Pipeline gồm các bước

1. **Làm sạch** (`data_loader`): loại người tích "từ giả" VCL (≈5.2k dòng, gian lận),
   loại tuổi phi lý → còn **34.576 / 39.775 dòng (86.9%)**.
2. **Tính điểm** (`scoring`): thang trả lời gốc 1–4 → trừ 1 về 0–3, cộng 14 câu/thang
   → điểm 0–42 → ánh xạ sang nhãn severity theo ngưỡng DASS-42 chuẩn.
   TIPI → 5 chiều Big Five (có reverse-scoring item đảo).
3. **Đặc trưng** (`features`): 7 đặc trưng số (age, familysize, 5 Big Five) +
   11 đặc trưng phân loại (gender, education, religion, race, …). Tiền xử lý nằm
   trong `sklearn.Pipeline` → **không rò rỉ** thống kê từ test sang train.
4. **Huấn luyện** (`model`): Dummy (baseline), Logistic Regression, Random Forest,
   XGBoost — đều **cân bằng lớp** (`class_weight` / `sample_weight`).
5. **Đánh giá** (`evaluate`): ưu tiên **macro-F1** và **balanced accuracy**
   (không chỉ accuracy thô vì lớp mất cân bằng).

## Kết quả (tập test 20%)

**Chế độ `binary` (khuyến nghị) — phân loại "nguy cơ cao":**

| Thang | Model tốt nhất | ROC-AUC | Balanced Acc | Baseline |
|-------|----------------|---------|--------------|----------|
| depression | Random Forest | 0.803 | ~0.73 | 0.50 |
| anxiety    | Logistic Reg. | 0.808 | ~0.73 | 0.50 |
| stress     | XGBoost       | 0.826 | ~0.75 | 0.50 |

**Chế độ `multiclass` (5 mức) — để đối chiếu:** macro-F1 0.34–0.38 (baseline 0.20).

**Đặc trưng quan trọng nhất ở cả 3 thang: `big5_emotional_stability`**
(đối nghịch với neuroticism) — hoàn toàn hợp lý về mặt tâm lý học. Theo sau là
extraversion, conscientiousness, tuổi.

### Đọc kết quả thế nào cho đúng

- Bài toán 5 lớp → đoán ngẫu nhiên cho balanced accuracy ≈ 0.20. Mô hình đạt
  **0.34–0.38**, tức **gần gấp đôi baseline** → có tín hiệu thật, nhưng **vừa phải**.
- Điều này **đúng như kỳ vọng**: nhân khẩu học + một bài tính cách 10 câu *không thể*
  thay thế bài DASS để chẩn đoán mức độ chính xác. Giá trị thực tế là **phân tầng rủi ro**
  (ai có khả năng cao thuộc nhóm nặng), không phải chẩn đoán lâm sàng.
- Confusion matrix cho thấy mô hình phân biệt tốt **Normal vs Extremely Severe**,
  nhưng lẫn giữa các mức kề nhau (Mild/Moderate/Severe) — đặc trưng của nhãn thứ bậc.

## Tuỳ chỉnh nhanh (đều ở `src/config.py`)

- `COLLAPSE_SEVERE = True` → gộp Severe + Extremely Severe thành 4 lớp như đề bài gốc.
- `MAX_FAKE_WORDS_ALLOWED` → nới/siết tiêu chí lọc gian lận.
- `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` → thêm/bớt đặc trưng.

## Hướng mở rộng

- **Hồi quy** điểm DASS liên tục (0–42) thay vì phân loại → giữ thông tin thứ bậc.
- **Ordinal classification** (mô hình tôn trọng thứ tự lớp) thay multiclass thường.
- Thêm `country` (gom top-N), `testelapse`, thời gian trả lời từng câu (`QnE`).
- Tinh chỉnh siêu tham số (Optuna / GridSearch) + hiệu chỉnh xác suất (calibration).

## Lưu ý đạo đức

Đây là dữ liệu khảo sát tự nguyện ẩn danh dùng cho mục đích học tập. Mô hình **không**
là công cụ chẩn đoán y tế và không nên dùng để ra quyết định về cá nhân thực.
