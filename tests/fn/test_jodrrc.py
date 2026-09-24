"""Tests for jodrrc.joseph_dirrec_strategy."""

from morie.fn import _array_core as np

from morie.fn.jodrrc import joseph_dirrec_strategy


def test_jodrrc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    result = joseph_dirrec_strategy(x, lags=[1, 2], horizon=3)
    assert isinstance(result, dict)
    assert 'forecast' in result
    assert len(result['forecast']) == 3


def test_jodrrc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 40)
    result = joseph_dirrec_strategy(x, lags=[1], horizon=1)
    assert isinstance(result, dict)
    assert 'forecast' in result
    assert len(result['forecast']) == 1
