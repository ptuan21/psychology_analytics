import numpy as np
import pandas as pd
import pytest

from src import nhanes


class TestNa:
    def test_replaces_refused_and_dont_know_codes(self):
        s = pd.Series([1, 7, 2, 9, 3])
        out = nhanes._na(s, [7, 9])
        assert out.tolist()[0] == 1
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == 2
        assert pd.isna(out.iloc[3])
        assert out.iloc[4] == 3

    def test_missing_column_scalar_input_returns_nan(self):
        # mô phỏng df.get("COL", np.nan) khi cột vắng mặt ở chu kỳ này
        assert np.isnan(nhanes._na(np.nan, [7, 9]))


class TestSmoke:
    def _demo(self, q20, q40) -> pd.DataFrame:
        return pd.DataFrame({"SMQ020": [q20], "SMQ040": [q40]})

    def test_never_smoked(self):
        out = nhanes._smoke(self._demo(2, np.nan))
        assert out.iloc[0] == 0

    def test_former_smoker(self):
        out = nhanes._smoke(self._demo(1, 3))
        assert out.iloc[0] == 1

    def test_current_smoker_every_day(self):
        out = nhanes._smoke(self._demo(1, 1))
        assert out.iloc[0] == 2

    def test_current_smoker_some_days(self):
        out = nhanes._smoke(self._demo(1, 2))
        assert out.iloc[0] == 2

    def test_missing_data_gives_nan(self):
        out = nhanes._smoke(self._demo(np.nan, np.nan))
        assert pd.isna(out.iloc[0])

    def test_refused_code_treated_as_missing(self):
        out = nhanes._smoke(self._demo(7, np.nan))
        assert pd.isna(out.iloc[0])


class TestMarital:
    def test_cycle_l_uses_dmdmartz(self):
        demo = pd.DataFrame({"DMDMARTZ": [1, 2, 3, 77, 99]})
        out = nhanes._marital(demo, "L")
        assert out.tolist()[:3] == ["partnered", "previously", "never"]
        assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])

    def test_other_cycles_use_dmdmartl_and_collapse_married_variants(self):
        demo = pd.DataFrame({"DMDMARTL": [1, 6, 2, 3, 4, 5]})
        out = nhanes._marital(demo, "J")
        assert out.tolist() == [
            "partnered", "partnered", "previously", "previously", "previously", "never",
        ]


class TestBuildCycle:
    def _demo(self, **overrides) -> pd.DataFrame:
        row = {
            "SEQN": [1, 2],
            "RIDAGEYR": [30, 70],
            "RIAGENDR": [1, 2],
            "DMDEDUC2": [3, 9],
            "RIDRETH1": [3, 4],
            "INDFMPIR": [1.5, 5.0],
            "WTMEC2YR": [1000.0, 2000.0],
            "SDMVPSU": [1, 2],
            "SDMVSTRA": [10, 20],
            "DMDMARTL": [1, 5],
            "SLD010H": [7, 7],
            "SMQ020": [2, 2],
            "SMQ040": [np.nan, np.nan],
            "FSDAD": [1, 2],
            "OCD150": [1, 3],
            "HIQ011": [1, 2],
        }
        row.update(overrides)
        return pd.DataFrame(row)

    def _dpq(self, seqn9=9) -> pd.DataFrame:
        # Người 1: tổng PHQ-9 = 1 (None-minimal). Người 2: một mục = seqn9 (9 = "không biết" -> thiếu).
        return pd.DataFrame({
            "SEQN": [1, 2],
            "DPQ010": [0, 3], "DPQ020": [1, 3], "DPQ030": [0, 3],
            "DPQ040": [0, 3], "DPQ050": [0, 3], "DPQ060": [0, 3],
            "DPQ070": [0, 3], "DPQ080": [0, 3], "DPQ090": [0, seqn9],
        })

    def _patch_read(self, monkeypatch, demo, dpq):
        def fake_read(comp, suf):
            if comp == "DEMO":
                return demo
            if comp == "DPQ":
                return dpq
            return None
        monkeypatch.setattr(nhanes, "_read", fake_read)

    def test_missing_core_component_returns_none(self, monkeypatch):
        monkeypatch.setattr(nhanes, "_read", lambda comp, suf: None)
        assert nhanes.build_cycle("J") is None

    def test_phq9_total_and_dep_level_for_complete_answers(self, monkeypatch):
        self._patch_read(monkeypatch, self._demo(), self._dpq())
        out = nhanes.build_cycle("J")
        row1 = out.loc[out["SEQN"] == 1].iloc[0]
        assert row1["phq9_total"] == 1
        assert row1["dep_level"] == "None-minimal"
        assert row1["dep_risk"] == 0

    def test_phq9_missing_item_yields_nan_total(self, monkeypatch):
        self._patch_read(monkeypatch, self._demo(), self._dpq(seqn9=9))
        out = nhanes.build_cycle("J")
        row2 = out.loc[out["SEQN"] == 2].iloc[0]
        assert pd.isna(row2["phq9_total"])
        assert pd.isna(row2["dep_level"])
        assert pd.isna(row2["dep_risk"])

    def test_dep_risk_true_when_total_at_least_ten(self, monkeypatch):
        dpq = pd.DataFrame({
            "SEQN": [1],
            "DPQ010": [2], "DPQ020": [2], "DPQ030": [2],
            "DPQ040": [2], "DPQ050": [2], "DPQ060": [0],
            "DPQ070": [0], "DPQ080": [0], "DPQ090": [0],
        })
        self._patch_read(monkeypatch, self._demo().iloc[[0]], dpq)
        out = nhanes.build_cycle("J")
        assert out.iloc[0]["phq9_total"] == 10
        assert out.iloc[0]["dep_risk"] == 1
        assert out.iloc[0]["dep_level"] == "Moderate"

    def test_sleep_variable_harmonized_across_cycles(self, monkeypatch):
        # Chu kỳ mới dùng SLD012, cũ dùng SLD010H. Cả hai bị clip ở 14 giờ.
        demo_new = self._demo(SLD012=[20, 6])
        self._patch_read(monkeypatch, demo_new, self._dpq())
        out_new = nhanes.build_cycle("J")
        assert out_new.loc[out_new["SEQN"] == 1, "sleep_hours"].iloc[0] == 14
        assert out_new.loc[out_new["SEQN"] == 2, "sleep_hours"].iloc[0] == 6

        demo_old = self._demo(SLD010H=[9, 6])
        self._patch_read(monkeypatch, demo_old, self._dpq())
        out_old = nhanes.build_cycle("H")
        assert out_old.loc[out_old["SEQN"] == 1, "sleep_hours"].iloc[0] == 9

    def test_age_group_binning(self, monkeypatch):
        self._patch_read(monkeypatch, self._demo(), self._dpq())
        out = nhanes.build_cycle("J")
        assert out.loc[out["SEQN"] == 1, "age_group"].iloc[0] == "26-35"
        assert out.loc[out["SEQN"] == 2, "age_group"].iloc[0] == "66+"

    def test_cycle_and_year_labels_attached(self, monkeypatch):
        self._patch_read(monkeypatch, self._demo(), self._dpq())
        out = nhanes.build_cycle("J")
        assert (out["cycle"] == "2017-2018").all()
        assert (out["year"] == 2017).all()


class TestWeightedPrevalence:
    def test_wprev_matches_manual_weighted_average(self):
        df = pd.DataFrame({"dep_risk": [1, 0, 1], "weight": [1.0, 1.0, 2.0]})
        assert nhanes.wprev(df) == pytest.approx(100 * (1 * 1 + 0 * 1 + 1 * 2) / 4)

    def test_wprev_empty_after_dropna_returns_nan(self):
        df = pd.DataFrame({"dep_risk": [np.nan], "weight": [1.0]})
        assert np.isnan(nhanes.wprev(df))

    def test_wprev_by_orders_groups_as_given(self):
        df = pd.DataFrame({
            "grp": ["a", "a", "b"],
            "dep_risk": [1, 0, 1],
            "weight": [1.0, 1.0, 1.0],
        })
        out = nhanes.wprev_by(df, "grp", ["b", "a"])
        assert list(out.index) == ["b", "a"]
        assert out["b"] == pytest.approx(100.0)
        assert out["a"] == pytest.approx(50.0)
