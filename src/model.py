"""
Định nghĩa mô hình và logic huấn luyện cho một target (1 thang DASS).

Xử lý mất cân bằng lớp:
  - LogisticRegression / RandomForest: class_weight="balanced"
  - XGBoost: sample_weight tính theo nghịch đảo tần suất lớp
Đánh giá ưu tiên macro-F1 & balanced accuracy (không chỉ accuracy thô).
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from .features import build_preprocessor


def make_models() -> dict:
    """Trả về dict {tên: estimator} (chưa gắn tiền xử lý)."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            multi_class="multinomial", random_state=C.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            n_jobs=-1, random_state=C.RANDOM_STATE,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", estimator),
    ])


def fit_pipeline(name: str, estimator, X_train, y_train) -> Pipeline:
    """Fit, có truyền sample_weight cho XGBoost để cân bằng lớp."""
    pipe = build_pipeline(estimator)
    if name == "xgboost":
        w = compute_sample_weight(class_weight="balanced", y=y_train)
        pipe.fit(X_train, y_train, clf__sample_weight=w)
    else:
        pipe.fit(X_train, y_train)
    return pipe
