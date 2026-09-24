"""Tests for netatp.network_attack_tolerance."""

from morie.fn import _array_core as np

from morie.fn.netatp import network_attack_tolerance


def test_netatp_basic():
    """Test basic functionality."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = network_attack_tolerance(A)
    assert isinstance(result, dict)
    assert "s_giant" in result


def test_netatp_edge():
    """Test edge cases."""
    A = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = network_attack_tolerance(A)
    assert isinstance(result, dict)
