"""Tests for relua.relu_activation."""

from morie.fn import _array_core as np

from morie.fn.relua import relu_activation


def test_relua_basic():
    """g(z) = max(0, z) with g'(z) = 1 for z > 0 and 0 otherwise."""
    z = [-2.0, -0.5, 0.0, 0.5, 3.0]
    result = relu_activation(np.array(z))
    assert [float(v) for v in result["activation"]] == [0.0, 0.0, 0.0, 0.5, 3.0]
    assert [float(v) for v in result["gradient"]] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert result["n"] == 5


def test_relua_edge():
    """The leaky variant keeps slope * z below zero (and slope as the
    gradient there); a single value works."""
    r = relu_activation(np.array([-2.0, 4.0]), slope=0.01)
    assert [float(v) for v in r["activation"]] == [-0.02, 4.0]
    assert [float(v) for v in r["gradient"]] == [0.01, 1.0]
    assert relu_activation(np.array([42.0]))["n"] == 1
