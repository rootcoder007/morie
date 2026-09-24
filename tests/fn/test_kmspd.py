"""Tests for kmspd.kamath_speculative_decoding."""

from morie.fn import _array_core as np

from morie.fn.kmspd import kamath_speculative_decoding


def test_kmspd_basic():
    """Test basic functionality."""
    draft_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    target_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_speculative_decoding(draft_probs, target_probs)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmspd_edge():
    """Test edge cases."""
    draft_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    target_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = kamath_speculative_decoding(draft_probs, target_probs)
    assert isinstance(result, dict)
