"""Tests for aitjsd.compositional_jsd."""

from morie.fn import _array_core as np

from morie.fn.aitjsd import compositional_jsd


def test_aitjsd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 1.0, 100)
    q = rng.uniform(0.0, 1.0, 100)
    result = compositional_jsd(p, q)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    value = result.get("estimate", result.get("statistic"))
    assert isinstance(value, float)
    assert value >= 0.0


def test_aitjsd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    p = rng.uniform(0.0, 1.0, 100)
    q = rng.uniform(0.0, 1.0, 100)
    result = compositional_jsd(p, q)
    assert isinstance(result, dict)
    value = result.get("estimate", result.get("statistic"))
    assert isinstance(value, float)
    assert value >= 0.0
