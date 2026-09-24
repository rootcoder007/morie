"""Tests for kmmedu.kamath_medusa_heads."""

from morie.fn import _array_core as np

from morie.fn.kmmedu import kamath_medusa_heads


def test_kmmedu_basic():
    """Test basic functionality."""
    hidden_state = 5
    medusa_heads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "tokens" in result


def test_kmmedu_edge():
    """Test edge cases."""
    hidden_state = 5
    medusa_heads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)
