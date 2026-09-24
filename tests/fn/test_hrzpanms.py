"""Tests for hrzpanms.horowitz_panel_max_score."""

from morie.fn import _array_core as np

from morie.fn.hrzpanms import horowitz_panel_max_score


def test_hrzpanms_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, T, d = 20, 5, 3
    x = rng.normal(0, 1, (n, T, d))
    y = rng.integers(0, 2, (n, T))
    result = horowitz_panel_max_score(x, y, n_periods=T, n_restarts=2)
    assert "beta" in result
    assert "score" in result
    assert "n_pairs" in result
    assert "n_discordant_pairs" in result
    assert "unidentified_columns" in result
    assert "intercept_identified" in result
    assert "smoothed" in result
    assert "bandwidth" in result
    assert "rate_exponent" in result
    assert "n" in result
    assert "T" in result
    assert "d" in result
    assert "method" in result


def test_hrzpanms_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, T, d = 10, 2, 2
    x = rng.normal(0, 1, (n, T, d))
    y = rng.integers(0, 2, (n, T))
    result = horowitz_panel_max_score(x, y, n_periods=T, n_restarts=2)
    assert "beta" in result
    assert "score" in result
    assert "n_pairs" in result
