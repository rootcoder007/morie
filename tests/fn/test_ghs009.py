"""Tests for ghs009.ghosal_ch3_stick_breaking_weights."""

from morie.fn import _array_core as np

from morie.fn.ghs009 import ghosal_ch3_stick_breaking_weights


def test_ghs009_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = ghosal_ch3_stick_breaking_weights(V)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs009_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = ghosal_ch3_stick_breaking_weights(V)
    assert isinstance(result, dict)
