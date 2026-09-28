"""Tests for morie.fn.isorg -- isotonic regression."""

from morie.fn import _array_core as np
from morie.fn.isorg import isorg, isotonic_regression


def test_isorg_monotone():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = isorg(x)
    assert r.name == "isotonic_regression"
    assert np.allclose(r.value, x)


def test_isorg_violation():
    x = np.array([1.0, 3.0, 2.0, 4.0])
    r = isorg(x)
    assert r.value[1] <= r.value[2] or np.isclose(r.value[1], r.value[2])


def test_isorg_alias():
    assert isorg is isotonic_regression


def test_isorg_pools_whole_blocks_as_isoreg():
    # stats::isoreg(c(3, 1, 2, 5, 4, 4.5, 2, 6))$yf: the block (5, 4, 4.5, 2) pools to its mean 3.875
    r = isotonic_regression([3.0, 1.0, 2.0, 5.0, 4.0, 4.5, 2.0, 6.0])
    assert [float(v) for v in r.value] == [2.0, 2.0, 2.0, 3.875, 3.875, 3.875, 3.875, 6.0]


def test_isorg_weighted_block_mean():
    r = isotonic_regression([3.0, 1.0, 5.0], w=[1.0, 3.0, 2.0])
    assert [float(v) for v in r.value] == [1.5, 1.5, 5.0]
