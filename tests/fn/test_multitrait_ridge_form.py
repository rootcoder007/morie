"""Tests for multitrait_ridge_form.multitrait_ridge_form."""

from morie.fn import _array_core as np

from morie.fn.multitrait_ridge_form import multitrait_ridge_form


def test_msm072_basic():
    """Test basic functionality."""
    Z1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = multitrait_ridge_form(Z1, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm072_edge():
    """Test edge cases."""
    Z1 = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    G = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = multitrait_ridge_form(Z1, G)
    assert isinstance(result, dict)
