"""Tests for morie.fn.sntag — sentence by age group."""

from morie.fn import _frame_core as pd

from morie.fn.sntag import sentence_by_age


class TestSentenceByAge:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_age(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "age_group" in result.columns

    def test_has_median(self, otis_df):
        result = sentence_by_age(otis_df)
        assert "median_sentence" in result.columns

    def test_n_groups(self, otis_df):
        result = sentence_by_age(otis_df)
        assert len(result) == otis_df["age_group"].nunique()
