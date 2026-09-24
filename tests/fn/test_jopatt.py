"""Tests for jopatt.joseph_patchtst."""

from morie.fn import _array_core as np

from morie.fn.jopatt import joseph_patchtst


def test_jopatt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # x must be a list of lists (2-D) to avoid boolean evaluation error in core function
    x = [list(rng.normal(0, 1, 50)) for _ in range(3)]
    patch_len = 8
    stride = 4
    transformer = 1e-5
    result = joseph_patchtst(x, patch_len, stride, transformer)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_jopatt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Minimal valid 2-D input
    x = [list(rng.normal(0, 1, 20)) for _ in range(2)]
    patch_len = 4
    stride = 2
    transformer = 1e-6
    result = joseph_patchtst(x, patch_len, stride, transformer)
    assert isinstance(result, dict)
