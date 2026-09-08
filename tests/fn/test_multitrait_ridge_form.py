"""Tests for multitrait_ridge_form.multitrait_ridge_form."""

from morie.fn import _array_core as np

from morie.fn.multitrait_ridge_form import multitrait_ridge_form


def test_msm072_basic():
    """Test basic functionality."""
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    third = np.random.default_rng(42).normal(0, 1, 100)
    argument = np.random.default_rng(42).normal(0, 1, 100)
    resCOV = np.random.default_rng(42).normal(0, 1, 100)
    result = multitrait_ridge_form(respectively, In, the, third, argument, resCOV)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm072_edge():
    """Test edge cases."""
    respectively = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    third = np.random.default_rng(42).normal(0, 1, 100)
    argument = np.random.default_rng(42).normal(0, 1, 100)
    resCOV = np.random.default_rng(42).normal(0, 1, 100)
    result = multitrait_ridge_form(respectively, In, the, third, argument, resCOV)
    assert isinstance(result, dict)
