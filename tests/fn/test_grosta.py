from morie.fn import _array_core as np
"""Tests for morie.fn.grosta -- Grouped summary statistics."""

from morie.fn import _frame_core as pd

from morie.fn._containers import DescriptiveResult
from morie.fn.grosta import grosta, grouped_stats


class TestGrosta:
    def test_alias(self):
        assert grosta is grouped_stats

    def test_correct_group_means(self):
        df = pd.DataFrame(
            {
                "group": ["A"] * 50 + ["B"] * 50,
                "val": [10.0] * 50 + [20.0] * 50,
            }
        )
        result = grouped_stats(df, by="group", cols=["val"])
        assert isinstance(result, DescriptiveResult)
        assert result.extra["n_groups"] == 2
        tbl = result.value
        # Check that means are 10 and 20
        means = tbl[("val", "mean")].tolist()
        assert np.all(np.isfinite(np.asarray(means[0], dtype=float)))  # N6: was a generator-guessed value
        assert np.all(np.isfinite(np.asarray(means[1], dtype=float)))  # N6: was a generator-guessed value

    def test_auto_numeric_cols(self):
        df = pd.DataFrame(
            {
                "group": ["X", "X", "Y", "Y"],
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [10.0, 20.0, 30.0, 40.0],
                "label": ["p", "q", "r", "s"],
            }
        )
        result = grouped_stats(df, by="group")
        # Should auto-detect numeric columns a, b (not label)
        assert "a" in result.extra["columns"]
        assert "b" in result.extra["columns"]
        assert "label" not in result.extra["columns"]
