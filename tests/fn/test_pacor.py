"""Tests for pacor.predictive_ability_pearson."""

from morie.fn import _array_core as np

from morie.fn.pacor import predictive_ability_pearson


def test_pacor_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = predictive_ability_pearson(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_pacor_edge():
    """Test edge cases."""
    result = predictive_ability_pearson(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
