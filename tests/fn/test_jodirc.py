"""Tests for jodirc.joseph_direct_multistep."""

from morie.fn import _array_core as np

from morie.fn.jodirc import joseph_direct_multistep


def test_jodirc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    result = joseph_direct_multistep(x, lags=[1, 2, 3], horizon=5)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_jodirc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 50)
    result = joseph_direct_multistep(x, lags=[1], horizon=1)
    assert isinstance(result, dict)
    assert len(result) > 0
