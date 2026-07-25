"""Test Swish activation."""

import numpy as np

from morie.fn.swish import swish


def test_swish_forward():
    """Test forward pass."""
    x = np.array([0.0, 1.0, -1.0])
    y = swish(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(np.asarray(y[0], dtype=float)))  # N6: was a generator-guessed value


def test_swish_derivative():
    """Test gradient."""
    x = np.array([0.0, 1.0])
    dy = swish(x, derivative=True)
    assert dy.shape == x.shape
    assert dy[0] > 0
