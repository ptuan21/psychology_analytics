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

---

# Phần mở rộng: Phân tích xu hướng trầm cảm thật (NHANES 2007–2023)

Ngoài dữ liệu DASS, dự án bổ sung **dữ liệu thật, đại diện dân số** từ **NHANES**
(CDC, Mỹ) để phân tích **xu hướng trầm cảm theo thời gian × tuổi × bậc học × nhiều yếu tố**
trong kỷ nguyên smartphone (2007 → nay), và định lượng yếu tố can thiệp được.

- **Nguồn:** 7 chu kỳ NHANES 2007–2023, tải trực tiếp từ CDC (công khai).
- **Quy mô:** 71.775 người (~37k có PHQ-9 hợp lệ); đo trầm cảm bằng **PHQ-9**.
- **Biến:** nhân khẩu (tuổi, giới, học vấn, chủng tộc, thu nhập, hôn nhân) +
  yếu tố can thiệp được (ngủ, rượu, vận động, ít vận động, hút thuốc, sức khỏe tự đánh giá).
- Mọi ước lượng **dùng trọng số khảo sát** (đại diện dân số).

## Quy trình (tái lập đầy đủ)

```bash
bash   scripts/fetch_nhanes.sh           # tải .xpt thật từ CDC -> data/nhanes/raw/
python scripts/build_nhanes.py           # hài hoà 7 chu kỳ -> data/nhanes/nhanes_pooled.csv
python scripts/analyze_nhanes_trends.py  # 6 biểu đồ xu hướng theo tuổi/bậc học/thời gian
python scripts/analyze_nhanes_factors.py # 4 biểu đồ theo giới/chủng tộc/thu nhập/...
python scripts/model_nhanes.py           # hồi quy logistic đa biến -> Odds Ratio
```

| File | Vai trò |
|------|---------|
| `src/nhanes.py` | logic hài hoà biến + tiện ích ước lượng có trọng số |
| `scripts/fetch_nhanes.sh` | tải dữ liệu thật |
| `scripts/build_nhanes.py` | dựng bảng gộp |
| `scripts/analyze_nhanes_trends.py` | xu hướng tuổi × bậc học × thời gian |
| `scripts/analyze_nhanes_factors.py` | mở rộng nhiều yếu tố |
| `scripts/model_nhanes.py` | hồi quy đa biến + kiểm định tương tác |

## Phát hiện chính (dữ liệu thật, có trọng số)

- **Tỉ lệ trầm cảm tăng** từ ~8% (2007–2018) lên **12.6% (2021–2023)**.
- **Tập trung ở người trẻ:** nhóm 18–25 từ ~7% → **19.8%**; mô hình tương tác xác nhận
  nguy cơ tăng **nhanh gấp 2.2×/thập kỷ** so với nhóm 36–50 (p≈5×10⁻⁷).
- **Yếu tố độc lập mạnh nhất (OR đã hiệu chỉnh):** ngủ <6h (2.53), sức khỏe kém (2.23),
  nữ giới (1.80), đang hút thuốc (1.68), ly hôn/goá (1.59). Thu nhập cao & tuổi ≥66 bảo vệ.
- → Đòn bẩy hỗ trợ: **giấc ngủ, cai thuốc, vận động, kết nối xã hội**, ưu tiên **giới trẻ**.

## Mô hình dự đoán (có giám sát, chia train/test)

`scripts/train_nhanes.py` huấn luyện mô hình dự đoán **nguy cơ trầm cảm** (PHQ-9 ≥ 10)
từ nhân khẩu + lối sống (`src/nhanes_model.py`), đánh giá theo 2 cách:

| Cách chia | Mô tả | AUC tốt nhất |
|-----------|-------|--------------|
| Ngẫu nhiên phân tầng 80/20 | năng lực dự đoán tổng quát | **0.80** (Random Forest) |
| Theo thời gian (học ≤2018, kiểm 2021–23) | học quá khứ, dự báo hiện tại | **0.75** (Logistic Reg.) |

- Mất cân bằng lớp (~10% dương) xử lý bằng `class_weight` / `scale_pos_weight`;
  báo cáo **ROC-AUC + PR-AUC** (phù hợp dữ liệu lệch).
- **Phát hiện:** AUC tụt ~0.05 khi kiểm theo thời gian → quan hệ **dịch chuyển hậu đại dịch**;
  mô hình tuyến tính đơn giản tổng quát bền hơn qua thời gian.
- Đầu ra: `roc_nhanes.png`, `roc_nhanes_temporal.png`, `cm_nhanes.png`,
  `importance_nhanes.png`, `metrics_nhanes.csv`.

### Cải tiến: tuning + calibration + chọn ngưỡng (`scripts/improve_nhanes.py`)

Tách **train 60% / valid 20% / test 20%** (chống rò rỉ): tinh chỉnh siêu tham số trên train,
**chọn ngưỡng trên valid**, báo cáo trên test.

- **(3) Tuning** (RandomizedSearchCV): XGBoost thắng (CV-AUC 0.794) → test AUC **0.806**.
- **(2) Calibration** (isotonic): xác suất đầu ra khớp tỉ lệ thực (Brier thấp hơn).
- **(1) Chọn ngưỡng** — bài học then chốt cho sàng lọc:

| Ngưỡng | Recall | Precision | % bị gắn cờ |
|--------|:---:|:---:|:---:|
| Mặc định 0.5 | **0.09** ❌ | 0.50 | 2% |
| Youden (0.08) | 0.80 | 0.20 | 39% |
| Recall ≥ 0.80 (0.08) | 0.81 | 0.19 | 42% |

→ Ngưỡng 0.5 **bỏ sót 91% ca** với dữ liệu lệch ~10%; ngưỡng sàng lọc bắt ~80% ca,
đổi lại cần theo dõi ~40% số người. Đây là đánh đổi đặc trưng của công cụ sàng lọc.
Đầu ra: `calibration_nhanes.png`, `threshold_nhanes.png`, `nhanes_thresholds.csv`.

## Giới hạn (trung thực)

- **Cắt ngang** (không theo dõi cùng người) → tương quan, **không nhân quả**.
- Trọng số cho ước lượng đại diện, nhưng **khoảng tin cậy OR là xấp xỉ** (chưa mô hình hoá
  đầy đủ thiết kế chọn mẫu phức tạp). Học vấn chỉ hỏi người ≥20; thiếu 2019–2020 (COVID).
- File `.xpt` thô bị `.gitignore` (tải lại bằng `fetch_nhanes.sh`); giữ `nhanes_pooled.csv`.

---

## Lưu ý đạo đức

Đây là dữ liệu khảo sát ẩn danh dùng cho mục đích học tập. Các mô hình **không**
là công cụ chẩn đoán y tế và không nên dùng để ra quyết định về cá nhân thực.
Xem thêm `data/codebook_extension.txt` (quy trình thu thập + an toàn) nếu mở rộng thu dữ liệu mới.
