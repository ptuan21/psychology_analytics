import numpy as np
import pandas as pd
import pytest

from src import evaluate


class TestDecisionCurvePoints:
    def test_treat_all_matches_prevalence_formula(self):
        y = np.array([1, 1, 0, 0, 0])  # prevalence = 0.4
        proba = np.array([0.9, 0.1, 0.9, 0.1, 0.5])
        dcp = evaluate.decision_curve_points(y, proba, thresholds=np.array([0.2, 0.5]))
        for _, row in dcp.iterrows():
            pt = row["threshold"]
            w = pt / (1 - pt)
            expected = 0.4 - 0.6 * w
            assert row["net_benefit_treat_all"] == pytest.approx(expected)

    def test_perfect_classifier_model_net_benefit_equals_prevalence(self):
        # Phân loại hoàn hảo: proba=1 cho dương, 0 cho âm -> ở mọi ngưỡng trong (0,1),
        # TP/n = prevalence, FP/n = 0 -> net_benefit_model = prevalence (không phụ thuộc pt).
        y = np.array([1, 1, 0, 0, 0, 0])  # prevalence = 1/3
        proba = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        dcp = evaluate.decision_curve_points(y, proba, thresholds=np.array([0.1, 0.5, 0.9]))
        assert dcp["net_benefit_model"].values == pytest.approx([1 / 3] * 3)

    def test_useless_model_never_beats_treat_none_by_much(self):
        # Mô hình ngẫu nhiên (proba không tách biệt lớp) -> net benefit model xấp xỉ 0
        # hoặc thấp hơn, không vượt trội rõ so với "không sàng lọc ai".
        rng_y = np.array([1, 0] * 50)
        proba = np.full(100, 0.5)  # mọi người cùng xác suất -> gắn cờ hết hoặc không ai
        dcp = evaluate.decision_curve_points(rng_y, proba, thresholds=np.array([0.3]))
        # tại pt=0.3 < 0.5, mọi người bị gắn cờ (proba>=pt) -> giống "sàng lọc tất cả"
        row = dcp.iloc[0]
        assert row["net_benefit_model"] == pytest.approx(row["net_benefit_treat_all"])

    def test_default_thresholds_span_almost_full_unit_interval(self):
        y = np.array([1, 0, 1, 0])
        proba = np.array([0.7, 0.3, 0.6, 0.4])
        dcp = evaluate.decision_curve_points(y, proba)
        assert dcp["threshold"].min() == pytest.approx(0.01)
        assert dcp["threshold"].max() == pytest.approx(0.99)
        assert len(dcp) == 99

    def test_columns_present(self):
        y = np.array([1, 0, 1, 0])
        proba = np.array([0.7, 0.3, 0.6, 0.4])
        dcp = evaluate.decision_curve_points(y, proba, thresholds=np.array([0.5]))
        assert set(dcp.columns) == {"threshold", "net_benefit_model", "net_benefit_treat_all"}
