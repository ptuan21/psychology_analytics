"""
Mô hình DỰ ĐOÁN (có giám sát) nguy cơ trầm cảm trên dữ liệu NHANES thật.

Mục tiêu: dep_risk (PHQ-9 ≥ 10) — phân loại nhị phân.
Đặc trưng: nhân khẩu + kinh tế xã hội + lối sống (KHÔNG dùng mục PHQ-9 -> tránh rò rỉ).
Thiếu dữ liệu được impute trong pipeline (giữ tối đa mẫu, không rò rỉ train->test).
Mất cân bằng lớp (~10% dương) xử lý bằng class_weight / scale_pos_weight.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C

TARGET = "dep_risk"
NUMERIC = ["age", "income_poverty", "sleep_hours", "sedentary_min", "gen_health", "year"]
CATEGORICAL = ["gender", "education_label", "race", "marital", "smoke_status"]


def select_xy(df: pd.DataFrame):
    """Lấy X (đặc trưng) và y (dep_risk) cho người lớn 18+ có nhãn hợp lệ."""
    d = df[(df["age"] >= 18) & df[TARGET].notna()].copy()
    X = d[NUMERIC + CATEGORICAL].copy()
    for c in CATEGORICAL:                      # mã số/đối tượng -> chuỗi, NaN giữ cho imputer
        X[c] = X[c].map(lambda v: np.nan if pd.isna(v) else str(v)).astype("object")
    y = d[TARGET].astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def make_models() -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=C.RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, class_weight="balanced",
            n_jobs=-1, random_state=C.RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=C.RANDOM_STATE),
    }


def fit_pipeline(name, est, X_tr, y_tr) -> Pipeline:
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", est)])
    if name == "xgboost":                      # cân bằng lớp cho XGB
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        pipe.named_steps["clf"].set_params(scale_pos_weight=spw)
    pipe.fit(X_tr, y_tr)
    return pipe


def feature_names(prep: ColumnTransformer) -> list:
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    return list(NUMERIC) + list(ohe.get_feature_names_out(CATEGORICAL))
