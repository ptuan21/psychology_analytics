#!/usr/bin/env bash
# Tải dữ liệu NHANES thật (CDC) cho các chu kỳ 2007–2023 về data/nhanes/raw/
# Công khai, không cần đăng ký. Dùng: bash scripts/fetch_nhanes.sh
set -u
cd "$(dirname "$0")/../data/nhanes" || exit 1
mkdir -p raw

# suffix:năm_bắt_đầu:nhãn_chu_kỳ
rows="E:2007:2007-2008 F:2009:2009-2010 G:2011:2011-2012 H:2013:2013-2014 I:2015:2015-2016 J:2017:2017-2018 L:2021:2021-2023"
# DEMO=nhân khẩu, DPQ=PHQ-9, SLQ=ngủ, ALQ=rượu, PAQ=vận động, SMQ=hút thuốc, HUQ=sức khoẻ tự đánh giá
# FSQ=an ninh lương thực, OCQ=việc làm, HIQ=bảo hiểm y tế (yếu tố xã hội)
comps="DEMO DPQ SLQ ALQ PAQ SMQ HUQ FSQ OCQ HIQ"

get(){
  comp=$1; suf=$2; start=$3; cyc=$4; out="raw/${comp}_${suf}.xpt"
  [ -f "$out" ] && [ -s "$out" ] && { echo "  ${comp}_${suf} cached"; return; }
  for url in \
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/$start/DataFiles/${comp}_${suf}.xpt" \
    "https://wwwn.cdc.gov/Nchs/Nhanes/$cyc/${comp}_${suf}.XPT" ; do
    code=$(curl -sS -L --max-time 120 -o "$out" "$url" -w "%{http_code}")
    if [ "$code" = "200" ] && [ -s "$out" ]; then echo "  ${comp}_${suf} OK"; return; fi
  done
  rm -f "$out"; echo "  ${comp}_${suf} --miss--"
}

for r in $rows; do
  suf="${r%%:*}"; rest="${r#*:}"; start="${rest%%:*}"; cyc="${rest#*:}"
  echo "== $cyc =="
  for comp in $comps; do get "$comp" "$suf" "$start" "$cyc"; done
done
echo "Xong. Tiếp theo: python3 scripts/build_nhanes.py"
