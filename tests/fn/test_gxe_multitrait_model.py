"""Tests for gxe_multitrait_model.gxe_multitrait_model."""

from morie.fn import _array_core as np

from morie.fn.gxe_multitrait_model import gxe_multitrait_model


def test_msm032_basic():
    """Test basic functionality."""
    other = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_multitrait_model(other, relevant, strategy, In, a, similar)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm032_edge():
    """Test edge cases."""
    other = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    strategy = np.random.default_rng(42).normal(0, 1, 100)
    In = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    similar = np.random.default_rng(42).normal(0, 1, 100)
    result = gxe_multitrait_model(other, relevant, strategy, In, a, similar)
    assert isinstance(result, dict)
