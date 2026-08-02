"""Tests for morie.fn.rskcl — risk classification."""

from morie.fn import _frame_core as pd

from morie.fn.rskcl import risk_classify


class TestRiskClassify:
    def test_returns_dataframe_with_risk_level(self, otis_df):
        result = risk_classify(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "risk_level" in result.columns

    def test_three_levels(self, otis_df):
        result = risk_classify(otis_df)
        levels = set(result["risk_level"].unique())
        assert levels == {"Low", "Medium", "High"}

    def test_same_length(self, otis_df):
        result = risk_classify(otis_df)
        assert len(result) == len(otis_df)
