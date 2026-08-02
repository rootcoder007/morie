"""Tests for morie.fn.sntmd — sentence by group."""

from morie.fn import _frame_core as pd

from morie.fn.sntmd import sentence_by_group


class TestSentenceByGroup:
    def test_returns_dataframe(self, otis_df):
        result = sentence_by_group(otis_df)
        assert isinstance(result, pd.DataFrame)
        assert "median_sentence" in result.columns

    def test_groups_match(self, otis_df):
        result = sentence_by_group(otis_df)
        assert len(result) == otis_df["region"].nunique()

    def test_n_sums(self, otis_df):
        result = sentence_by_group(otis_df)
        assert result["n"].sum() == len(otis_df)
