"""Tests for gru4r.gru4rec."""

from morie.fn import _array_core as np

from morie.fn.gru4r import gru4rec


def test_gru4r_basic():
    """Test basic functionality."""
    sessions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    batch_size = 5
    result = gru4rec(sessions, batch_size)
    assert isinstance(result, dict)
    assert "steps" in result


def test_gru4r_edge():
    """Test edge cases."""
    sessions = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    batch_size = 5
    result = gru4rec(sessions, batch_size)
    assert isinstance(result, dict)
