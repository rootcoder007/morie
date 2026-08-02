"""Tests for hmosf.geron_one_shot."""

from morie.fn import _array_core as np

from morie.fn.hmosf import geron_one_shot


def test_hmosf_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    example = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_shot(model, example, query)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmosf_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    example = np.random.default_rng(42).normal(0, 1, 100)
    query = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_one_shot(model, example, query)
    assert isinstance(result, dict)
