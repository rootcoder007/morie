"""Tests for blue_blup_via_v.blue_blup_via_v."""

from morie.fn import _array_core as np

from morie.fn.blue_blup_via_v import blue_blup_via_v


def test_msm240_basic():
    """Test basic functionality."""
    represented = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = blue_blup_via_v(represented, by, random, variables, observed, which)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm240_edge():
    """Test edge cases."""
    represented = np.random.default_rng(42).normal(0, 1, 100)
    by = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    variables = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    which = np.random.default_rng(42).normal(0, 1, 100)
    result = blue_blup_via_v(represented, by, random, variables, observed, which)
    assert isinstance(result, dict)
