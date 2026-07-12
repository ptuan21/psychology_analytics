import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import scoring


def _dass_row(answer: int) -> dict:
    """Trả về dict Q1A..Q42A cùng một giá trị (1..4) cho mọi câu."""
    return {f"Q{i}A": answer for i in range(1, 43)}


class TestDassScores:
    def test_all_minimum_answers_score_zero(self):
        df = pd.DataFrame([_dass_row(1)])
        out = scoring.dass_scores(df)
        assert out["depression_score"].iloc[0] == 0
        assert out["anxiety_score"].iloc[0] == 0
        assert out["stress_score"].iloc[0] == 0

    def test_all_maximum_answers_score_full_range(self):
        df = pd.DataFrame([_dass_row(4)])
        out = scoring.dass_scores(df)
        # 14 câu, mỗi câu (4-1)=3 -> 42
        assert out["depression_score"].iloc[0] == 42
        assert out["anxiety_score"].iloc[0] == 42
        assert out["stress_score"].iloc[0] == 42

    def test_does_not_mutate_input(self):
        df = pd.DataFrame([_dass_row(2)])
        scoring.dass_scores(df)
        assert "depression_score" not in df.columns

    def test_only_uses_items_from_own_subscale(self):
        # Câu depression = 4, mọi câu khác = 1 -> chỉ depression_score bị ảnh hưởng.
        row = _dass_row(1)
        row["Q3A"] = 4  # Q3 thuộc depression
        df = pd.DataFrame([row])
        out = scoring.dass_scores(df)
        assert out["depression_score"].iloc[0] == 3  # (4-1) + 13*(1-1)
        assert out["anxiety_score"].iloc[0] == 0
        assert out["stress_score"].iloc[0] == 0


class TestSeverityCutoffs:
    @pytest.mark.parametrize("score,expected", [
        (0, "Normal"), (9, "Normal"),
        (10, "Mild"), (13, "Mild"),
        (14, "Moderate"), (20, "Moderate"),
        (21, "Severe"), (27, "Severe"),
        (28, "Extremely Severe"), (42, "Extremely Severe"),
    ])
    def test_depression_boundaries(self, score, expected):
        assert scoring._to_severity(score, C.SEVERITY_CUTOFFS["depression"]) == expected

    def test_out_of_range_falls_back_to_last_label(self):
        # score vượt mọi cutoff (không nên xảy ra với dữ liệu hợp lệ 0..42)
        assert scoring._to_severity(999, C.SEVERITY_CUTOFFS["depression"]) == "Extremely Severe"


class TestDassSeverity:
    def _scored_df(self, depression_score: int) -> pd.DataFrame:
        return pd.DataFrame({
            "depression_score": [depression_score],
            "anxiety_score": [0],
            "stress_score": [0],
        })

    def test_assigns_level_from_score(self, monkeypatch):
        monkeypatch.setattr(C, "COLLAPSE_SEVERE", False)
        out = scoring.dass_severity(self._scored_df(30))
        assert out["depression_level"].iloc[0] == "Extremely Severe"

    def test_collapse_severe_merges_top_two_levels(self, monkeypatch):
        monkeypatch.setattr(C, "COLLAPSE_SEVERE", True)
        out = scoring.dass_severity(self._scored_df(30))
        assert out["depression_level"].iloc[0] == "Severe"

    def test_no_collapse_keeps_extremely_severe(self, monkeypatch):
        monkeypatch.setattr(C, "COLLAPSE_SEVERE", False)
        out = scoring.dass_severity(self._scored_df(30))
        assert out["depression_level"].iloc[0] != "Severe"


class TestBigFive:
    def _tipi_row(self, **overrides) -> dict:
        row = {f"TIPI{i}": 4 for i in range(1, 11)}
        row.update(overrides)
        return row

    def test_reverse_scored_item_is_inverted(self):
        # extraversion = mean(TIPI1, 8 - TIPI6)
        df = pd.DataFrame([self._tipi_row(TIPI1=7, TIPI6=1)])
        out = scoring.big_five(df)
        assert out["big5_extraversion"].iloc[0] == pytest.approx((7 + (8 - 1)) / 2.0)

    def test_missing_item_propagates_nan(self):
        df = pd.DataFrame([self._tipi_row(TIPI1=np.nan)])
        out = scoring.big_five(df)
        assert pd.isna(out["big5_extraversion"].iloc[0])

    def test_all_midpoint_answers_give_midpoint_score(self):
        # thang 1..7, mọi câu = 4 -> mean(4, 8-4) = 4
        df = pd.DataFrame([self._tipi_row()])
        out = scoring.big_five(df)
        for dim in C.BIG_FIVE:
            assert out[f"big5_{dim}"].iloc[0] == pytest.approx(4.0)


class TestMakeTarget:
    def _level_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "depression_level": ["Normal", "Mild", "Moderate", "Severe"],
            "depression_score": [0, 12, 14, 30],
        })

    def test_multiclass_codes_follow_severity_order(self):
        y, labels, kind = scoring.make_target(self._level_df(), "depression", "multiclass")
        assert kind == "clf"
        assert labels == ["Normal", "Mild", "Moderate", "Severe"]
        assert list(y) == [0, 1, 2, 3]

    def test_multiclass_excludes_absent_labels(self):
        df = self._level_df()
        y, labels, kind = scoring.make_target(df, "depression", "multiclass")
        assert "Extremely Severe" not in labels

    def test_binary_uses_high_risk_cutoff(self):
        y, labels, kind = scoring.make_target(self._level_df(), "depression", "binary")
        assert kind == "clf"
        assert labels == C.BINARY_LABELS
        # cutoff depression = 14 -> score>=14 là nguy cơ cao
        assert list(y) == [0, 0, 1, 1]

    def test_regression_returns_float_score_and_no_labels(self):
        y, labels, kind = scoring.make_target(self._level_df(), "depression", "regression")
        assert kind == "reg"
        assert labels is None
        assert list(y) == [0.0, 12.0, 14.0, 30.0]
        assert y.dtype == float

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            scoring.make_target(self._level_df(), "depression", "not_a_mode")
