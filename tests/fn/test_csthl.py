"""Tests for morie.fn.csthl — custody health access."""

from morie.fn import _frame_core as pd

from morie.fn.csthl import custody_health_access


class TestCustodyHealthAccess:
    def test_returns_dataframe(self, otis_df):
        result = custody_health_access(otis_df)
        assert isinstance(result, pd.DataFrame)

    def test_rate_bounded(self, otis_df):
        result = custody_health_access(otis_df)
        assert (result["rate"] >= 0).all()
        assert (result["rate"] <= 1).all()

    def test_groups_match(self, otis_df):
        result = custody_health_access(otis_df)
        assert set(result["region"]) == set(otis_df["region"].unique())

    def test_custom_cols(self, otis_df):
        result = custody_health_access(otis_df, group_col="facility_type")
        assert len(result) > 0
