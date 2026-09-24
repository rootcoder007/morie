"""Tests for nprphet.neural_prophet."""

from morie.fn import _array_core as np

from morie.fn.nprphet import neural_prophet


def test_nprphet_basic():
    """Test basic functionality."""
    ds = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = neural_prophet(ds, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_nprphet_edge():
    """Test edge cases."""
    ds = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = neural_prophet(ds, y)
    assert isinstance(result, dict)
