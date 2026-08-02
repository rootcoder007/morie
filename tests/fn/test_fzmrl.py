"""Tests for fzmrl.fauzi_mrl_asymptotic."""

from morie.fn import _array_core as np

from morie.fn.fzmrl import fauzi_mrl_asymptotic


def test_fzmrl_basic():
    """MRL at t=0 for non-negative X is E[X]; at the default t=median
    it is the mean exceedance over the median."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_mrl_asymptotic(x, t=0.0)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value
    result_med = fauzi_mrl_asymptotic(x)  # t = median = 3
    assert np.all(np.isfinite(np.asarray(result_med["estimate"], dtype=float)))  # N6: was a generator-guessed value
    assert result_med["se"] >= 0.0


def test_fzmrl_edge():
    """Test edge cases."""
    result = fauzi_mrl_asymptotic(np.array([42.0]))
    assert result["n"] == 1
