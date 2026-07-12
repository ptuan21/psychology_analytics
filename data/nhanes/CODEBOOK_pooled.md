# Data dictionary — `nhanes_pooled.csv`

Bảng gộp & hài hoà từ **NHANES** (CDC, Mỹ), 7 chu kỳ **2007–2023**, một dòng / người.
Tạo lại bằng: `bash scripts/fetch_nhanes.sh && python scripts/build_nhanes.py`
(logic ở `src/nhanes.py`). Mỗi dòng là một người tham gia khảo sát thật.

**Nguồn gốc & giấy phép:** NHANES là dữ liệu **công khai (public domain)** của chính phủ
Mỹ (NCHS/CDC). File `.xpt` gốc tải từ `wwwn.cdc.gov`. Có thể dùng lại tự do cho nghiên cứu.

> Muốn ước lượng **đại diện dân số** phải dùng cột `weight` (+ `psu`, `strata`).
> Trầm cảm đo bằng PHQ-9 (sàng lọc, không phải chẩn đoán). Học vấn chỉ hỏi người ≥20 tuổi.
> 2019–2020 thiếu (gộp do gián đoạn COVID).

| Cột | Kiểu | Mô tả & mã hoá |
|-----|------|----------------|
| `SEQN` | int | Mã định danh người tham gia (theo từng chu kỳ NHANES) |
| `cycle` | str | Chu kỳ khảo sát, vd `2007-2008` … `2021-2023` |
| `year` | int | Năm bắt đầu chu kỳ (2007, 2009, …, 2021) |
| `age` | num | Tuổi (năm); ≥80 bị top-code theo NHANES |
| `age_group` | str | `12-17, 18-25, 26-35, 36-50, 51-65, 66+` |
| `gender` | str | `Male` / `Female` |
| `education` | num | Mã DMDEDUC2 (1–5); chỉ người ≥20 |
| `education_label` | str | `<9th, 9-11th, HS/GED, Some college, College grad` |
| `race` | num | RIDRETH1: 1=Mexican-Am, 2=Other Hispanic, 3=White, 4=Black, 5=Other/Multi |
| `income_poverty` | num | Tỉ lệ thu nhập gia đình / ngưỡng nghèo (0–5; 5 = top-code) |
| `marital` | str | `partnered` / `previously` (goá/ly hôn/ly thân) / `never` |
| `weight` | num | Trọng số khảo sát WTMEC2YR (dùng cho ước lượng dân số) |
| `psu` | num | Đơn vị chọn mẫu sơ cấp SDMVPSU (cho SE design-based) |
| `strata` | num | Tầng chọn mẫu SDMVSTRA |
| `phq9_total` | num | Tổng PHQ-9 (0–27); NaN nếu thiếu mục |
| `dep_level` | str | `None-minimal, Mild, Moderate, Mod-severe, Severe` |
| `dep_risk` | bool | **Biến mục tiêu**: PHQ-9 ≥ 10 (trầm cảm có ý nghĩa lâm sàng) |
| `sleep_hours` | num | Số giờ ngủ/đêm trung bình |
| `alc_drinks_day` | num | Số ly rượu/ngày uống điển hình (người có uống) |
| `phys_moderate` | num | Có vận động vừa hằng tuần? 1=Có, 2=Không (thiếu ở 2021–2023) |
| `sedentary_min` | num | Số phút ngồi/ít vận động mỗi ngày |
| `smoke_status` | num | 0=Chưa từng, 1=Đã bỏ, 2=Đang hút |
| `gen_health` | num | Sức khoẻ tự đánh giá: 1=Excellent … 5=Poor |
| `food_security` | num | An ninh lương thực (người lớn): 1=Đủ ăn … 4=Rất thiếu |
| `food_insecure` | num | 1 nếu `food_security` ∈ {3,4}, else 0 |
| `employment` | str | `employed` / `unemployed` / `not_in_labor` |
| `insured` | num | Có bảo hiểm y tế? 1=Có, 0=Không |

**Độ phủ (tỉ lệ không thiếu, xấp xỉ):** phq9 51%, education 59% (≥20 tuổi), sleep 65%,
sedentary 69%, smoke 61%, gen_health ~100%, food_security 96%, employment 66%, insured 100%.
