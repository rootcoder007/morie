"""Test Mish activation."""

from morie.fn import _array_core as np

from morie.fn.mish import mish


def test_mish_forward():
    """Test forward pass."""
    x = np.array([0.0, 1.0, -1.0])
    y = mish(x)
    assert y.shape == x.shape


def test_mish_derivative():
    """Test gradient."""
    x = np.array([0.0, 1.0])
    dy = mish(x, derivative=True)
    assert dy.shape == x.shape
