"""Tests for kmptun.kamath_prompt_tuning."""

from morie.fn import _array_core as np

from morie.fn.kmptun import kamath_prompt_tuning


def test_kmptun_basic():
    """Test basic functionality."""
    P = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_prompt_tuning(P, X)
    assert isinstance(result, dict)
    assert "estimate" in result or "X_aug" in result


def test_kmptun_edge():
    """Test edge cases."""
    P = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kamath_prompt_tuning(P, X)
    assert isinstance(result, dict)
