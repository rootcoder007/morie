"""Tests for wsmadm.wasserman_admissible."""

from morie.fn import _array_core as np

from morie.fn.wsmadm import wasserman_admissible


def test_wsmadm_basic():
    """Test basic functionality."""
    risk = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_admissible(risk)
    assert isinstance(result, dict)
    assert "admissible" in result


def test_wsmadm_edge():
    """Test edge cases."""
    risk = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_admissible(risk)
    assert isinstance(result, dict)
