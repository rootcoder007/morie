"""Tests for morie.fn.winmea -- Winsorized mean."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._containers import DescriptiveResult
from morie.fn.winmea import winmea, winsorized_mean


class TestWinmea:
    def test_alias(self):
        assert winmea is winsorized_mean

    def test_no_trim(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = winsorized_mean(df, col="x", proportion=0.0)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 3.0) < 1e-10

    def test_trims_outliers(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        df = pd.DataFrame({"x": x})
        result = winsorized_mean(df, col="x", proportion=0.1)
        assert result.value < np.mean(x)


def test_winsorized_mean_and_variance_recomputed():

    import pytest

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    k = 2
    s = sorted(x)
    w = [s[k]] * k + s[k : 10 - k] + [s[10 - k - 1]] * k
    m = sum(w) / 10
    r = winsorized_mean(np.array(x), proportion=0.2)
    assert r.value == pytest.approx(m, rel=1e-14)
    assert r.extra["winsorized_variance"] == pytest.approx(sum((v - m) ** 2 for v in w) / 9, rel=1e-13)
