"""Tests for hrzt41.horowitz_thm4_1_id_median."""

from morie.fn import _array_core as np

from morie.fn.hrzt41 import horowitz_thm4_1_id_median


def test_hrzt41_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_thm4_1_id_median(x, beta)
    assert isinstance(result, dict)
    assert "identified" in result


def test_hrzt41_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    beta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = horowitz_thm4_1_id_median(x, beta)
    assert isinstance(result, dict)
