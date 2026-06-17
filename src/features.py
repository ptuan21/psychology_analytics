"""
Xây ma trận đặc trưng X và pipeline tiền xử lý.

Đặc trưng = nhân khẩu học + Big Five (KHÔNG có câu DASS).
Pipeline tiền xử lý (đặt trong sklearn Pipeline để tránh rò rỉ qua train/test):
  - số:        impute median -> chuẩn hoá (StandardScaler)
  - phân loại: 0/NaN -> hạng mục "missing" -> One-Hot
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def _cat_to_str(col: pd.Series) -> pd.Series:
    """0 -> NaN, còn lại -> chuỗi mã (vd 3.0 -> '3'); NaN giữ nguyên cho imputer."""
    col = col.replace(0, np.nan)
    return col.map(lambda v: np.nan if pd.isna(v) else str(int(v))).astype("object")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lấy đúng các cột đặc trưng. Phân loại -> chuỗi mã, 0/NaN sẽ thành 'missing'."""
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for c in C.CATEGORICAL_FEATURES:
        X[c] = _cat_to_str(X[c])
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", numeric, C.NUMERIC_FEATURES),
        ("cat", categorical, C.CATEGORICAL_FEATURES),
    ])


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Tên đặc trưng sau One-Hot (gọi sau khi fit)."""
    names = list(C.NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names += list(ohe.get_feature_names_out(C.CATEGORICAL_FEATURES))
    return names
