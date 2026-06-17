"""
GIẢI THÍCH mô hình (SHAP) — vì sao mô hình gắn cờ một người là nguy cơ trầm cảm.

Dùng TreeExplainer trên XGBoost (huấn luyện trên đặc trưng đã tiền xử lý) để có:
  - Giải thích TOÀN CỤC: yếu tố nào đẩy dự đoán nhiều nhất (bar + beeswarm).
  - Giải thích CỤC BỘ: với từng cá nhân, yếu tố nào kéo nguy cơ lên/xuống (waterfall).
  - Dependence: ảnh hưởng của 1 yếu tố thay đổi theo giá trị của nó.

Đầu ra (outputs/figures/nhanes/):
  shap_bar.png, shap_beeswarm.png, shap_dependence.png,
  shap_case_high1.png, shap_case_high2.png

Chạy:  python3 scripts/explain_nhanes.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import nhanes, nhanes_model as nm

FIG = C.FIG_DIR / "nhanes"


def _save(name):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(FIG / name, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  + {name}")


def main():
    X, y = nm.select_xy(nhanes.load_pooled())
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)

    # Tiền xử lý TÁCH RIÊNG để SHAP giải thích trên đặc trưng booster thực sự thấy
    prep = nm.build_preprocessor()
    Xtr_t = prep.fit_transform(Xtr)
    Xte_t = prep.transform(Xte)
    # XGBoost cấm '<' '[' ']' trong tên đặc trưng -> làm sạch
    clean = lambda s: (s.replace("<", "lt").replace(">", "gt")
                       .replace("[", "(").replace("]", ")"))
    names = [clean(n) for n in nm.feature_names(prep)]
    to_df = lambda M: pd.DataFrame(M.toarray() if hasattr(M, "toarray") else M, columns=names)
    df_te = to_df(Xte_t)

    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
        random_state=C.RANDOM_STATE,
        scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model.fit(to_df(Xtr_t), ytr)

    explainer = shap.TreeExplainer(model)
    exp = explainer(df_te)                       # Explanation (log-odds lớp dương)
    print(f"Giải thích {len(df_te):,} ca test\n")

    # ---- toàn cục ----
    shap.plots.bar(exp, max_display=15, show=False)
    plt.title("SHAP — yếu tố ảnh hưởng nhất (trung bình |tác động|)")
    _save("shap_bar.png")

    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP beeswarm — chiều & độ lớn tác động theo từng người")
    _save("shap_beeswarm.png")

    # yếu tố quan trọng nhất -> dependence plot
    top = names[int(np.argmax(np.abs(exp.values).mean(0)))]
    shap.plots.scatter(exp[:, top], show=False)
    plt.title(f"SHAP dependence — {top}")
    _save("shap_dependence.png")

    # ---- cục bộ: 2 ca bị gắn cờ rủi ro cao nhất ----
    proba = model.predict_proba(df_te)[:, 1]
    order = np.argsort(proba)[::-1]
    feat_cols = nm.NUMERIC + nm.CATEGORICAL
    for k, idx in enumerate(order[:2], 1):
        row = Xte.iloc[idx]
        print(f"Ca rủi ro cao #{k}: xác suất={proba[idx]:.2f} | "
              + ", ".join(f"{c}={row[c]}" for c in feat_cols if pd.notna(row[c])))
        shap.plots.waterfall(exp[idx], max_display=12, show=False)
        plt.title(f"Vì sao gắn cờ — ca #{k} (P={proba[idx]:.2f})")
        _save(f"shap_case_high{k}.png")


if __name__ == "__main__":
    main()
