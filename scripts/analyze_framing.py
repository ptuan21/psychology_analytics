"""
Phân tích vì sao macro-F1 ~0.35 và liệu mô hình có thực sự "kém" không.

Kiểm 3 cách ĐÓNG KHUNG LẠI bài toán (cùng đặc trưng, cùng dữ liệu):
  1. Adjacent accuracy: dự đoán có lệch <= 1 mức so với thực tế không?
     (vì Mild/Moderate/Severe là các lát cắt nhân tạo trên 1 thang liên tục)
  2. Nhị phân "nguy cơ cao" (Moderate trở lên) -> đo ROC-AUC.
  3. Hồi quy điểm liên tục 0..42 -> đo MAE & Spearman.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import data_loader, features, scoring
from src.features import build_preprocessor
from sklearn.pipeline import Pipeline

df = scoring.build_targets_and_features(data_loader.load_clean(verbose=False))
X = features.select_features(df)

print(f"{'thang':<11} {'adj_acc(±1)':>11} {'binary_AUC':>11} {'reg_MAE':>9} {'reg_R²':>8} {'spearman':>9}")
print("-" * 62)
for t in C.DASS_SUBSCALES:
    labels = [l for l in C.SEVERITY_ORDER if l in set(df[f"{t}_level"])]
    code = {lab: i for i, lab in enumerate(labels)}
    y_ord = df[f"{t}_level"].map(code).astype(int)       # 0..4 (thứ bậc)
    y_bin = (df[f"{t}_score"] >= dict(depression=14, anxiety=10, stress=19)[t]).astype(int)
    y_reg = df[f"{t}_score"].astype(float)               # điểm liên tục 0..42

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=C.TEST_SIZE,
        stratify=y_ord, random_state=C.RANDOM_STATE)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]

    # 1) multiclass -> adjacent accuracy
    clf = Pipeline([("prep", build_preprocessor()),
                    ("clf", RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5,
                        class_weight="balanced", n_jobs=-1,
                        random_state=C.RANDOM_STATE))])
    clf.fit(Xtr, y_ord.iloc[idx_tr])
    pred_ord = clf.predict(Xte)
    adj = np.mean(np.abs(pred_ord - y_ord.iloc[idx_te].values) <= 1)

    # 2) nhị phân -> AUC
    clfb = Pipeline([("prep", build_preprocessor()),
                     ("clf", RandomForestClassifier(
                         n_estimators=300, min_samples_leaf=5,
                         class_weight="balanced", n_jobs=-1,
                         random_state=C.RANDOM_STATE))])
    clfb.fit(Xtr, y_bin.iloc[idx_tr])
    proba = clfb.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_bin.iloc[idx_te], proba)

    # 3) hồi quy -> MAE, R², Spearman
    reg = Pipeline([("prep", build_preprocessor()),
                    ("reg", RandomForestRegressor(
                        n_estimators=300, min_samples_leaf=5,
                        n_jobs=-1, random_state=C.RANDOM_STATE))])
    reg.fit(Xtr, y_reg.iloc[idx_tr])
    yhat = reg.predict(Xte)
    mae = mean_absolute_error(y_reg.iloc[idx_te], yhat)
    r2 = reg.score(Xte, y_reg.iloc[idx_te])
    rho = spearmanr(yhat, y_reg.iloc[idx_te]).correlation

    print(f"{t:<11} {adj:>11.3f} {auc:>11.3f} {mae:>9.2f} {r2:>8.3f} {rho:>9.3f}")
