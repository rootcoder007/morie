"""Tests for extended_predictor.extended_predictor."""

from morie.fn import _array_core as np

from morie.fn.extended_predictor import extended_predictor


def test_msm061_basic():
    """Test basic functionality."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    parameter = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    result = extended_predictor(j, inverse, of, the, parameter, any)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_msm061_edge():
    """Test edge cases."""
    j = np.random.default_rng(42).normal(0, 1, 100)
    inverse = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    parameter = np.random.default_rng(42).normal(0, 1, 100)
    any = np.random.default_rng(42).normal(0, 1, 100)
    result = extended_predictor(j, inverse, of, the, parameter, any)
    assert isinstance(result, dict)
