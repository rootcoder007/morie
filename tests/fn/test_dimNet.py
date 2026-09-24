"""Tests for dimNet.dimenet."""

from morie.fn import _array_core as np
from morie.fn.dimNet import angle_between
import math


def test_dimNet_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r_i = rng.normal(0, 1, 3)
    r_j = rng.normal(0, 1, 3)
    r_k = rng.normal(0, 1, 3)
    result = angle_between(r_i, r_j, r_k)
    assert math.isfinite(result)
    assert 0 <= result <= math.pi


def test_dimNet_edge():
    """Test edge cases."""
    r_i = np.array([0.0, 0.0, 0.0])
    r_j = np.array([1.0, 0.0, 0.0])
    r_k = np.array([-1.0, 0.0, 0.0])
    result = angle_between(r_i, r_j, r_k)
    assert math.isfinite(result)
    assert 0 <= result <= math.pi
