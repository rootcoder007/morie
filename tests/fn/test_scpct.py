"""Tests for scpct -- percentile norms."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.scpct import percentile_norms


class TestPercentileNorms:
    def test_basic(self):
        scores = np.arange(1, 101, dtype=float)
        result = percentile_norms(scores)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value) == 9

    def test_median_at_50(self):
        scores = np.arange(1, 101, dtype=float)
        result = percentile_norms(scores, percentiles=[50])
        assert abs(result.value.iloc[0]["score_value"] - 50.5) < 1


def test_percentile_table_recomputed():
    import pytest

    s = [12.0, 15.0, 9.0, 20.0, 17.0, 11.0, 14.0]
    xs = sorted(s)

    def q(p):
        h = 6 * p / 100
        lo = int(h)
        return xs[lo] + (h - lo) * (xs[min(lo + 1, 6)] - xs[lo])

    r = percentile_norms(s, percentiles=[10, 50, 90])
    assert [float(v) for v in r.value["score_value"]] == pytest.approx([q(10), q(50), q(90)], rel=1e-13)
