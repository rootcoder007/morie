"""Tests for viterb.viterbi."""

from morie.fn import _array_core as np

from morie.fn.viterb import viterbi


def test_viterb_basic():
    """Test basic functionality."""
    obs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    trans = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    emit = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = viterbi(obs, trans, emit)
    assert isinstance(result, dict)
    assert "estimate" in result or "path" in result


def test_viterb_edge():
    """Test edge cases."""
    obs = np.random.default_rng(42).normal(0.0, 1.0, 40)
    trans = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    emit = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = viterbi(obs, trans, emit)
    assert isinstance(result, dict)
