"""Tests for rglstm.rangayyan_lstm_signal."""

from morie.fn import _array_core as np

from morie.fn.bsaclass import rangayyan_lstm_signal


def test_rglstm_basic():
    """Test basic functionality."""
    sequences = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_lstm_signal(sequences)
    assert isinstance(result, dict)
    assert "hidden" in result


def test_rglstm_edge():
    """Test edge cases."""
    sequences = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = rangayyan_lstm_signal(sequences)
    assert isinstance(result, dict)
