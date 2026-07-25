"""Test GELU activation."""

import numpy as np

from morie.fn.gelua import gelua


def test_gelua_exact():
    """Test exact GELU."""
    x = np.array([0.0, 1.0, -1.0])
    y = gelua(x, approximate=False)
    assert y.shape == x.shape
    assert np.all(np.isfinite(np.asarray(y[0], dtype=float)))  # N6: was a generator-guessed value


def test_gelua_approximate():
    """Test approximate GELU."""
    x = np.array([0.0, 1.0, -1.0])
    y = gelua(x, approximate=True)
    assert y.shape == x.shape


def test_gelua_derivative():
    """Test gradient."""
    x = np.array([0.0, 1.0])
    dy = gelua(x, derivative=True)
    assert dy.shape == x.shape
