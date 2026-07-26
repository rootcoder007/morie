"""rslnk: residual / skip connection.  y = x + F(x)  (He et al. 2016)."""

import numpy as np
import pytest

from morie.fn.rslnk import residual_connection as res


def test_rslnk_adds_the_branch_to_the_input():
    x = np.array([1.0, 2.0, 3.0])
    fx = np.array([0.5, -1.0, 2.0])
    assert np.asarray(res(x, fx)["y"]) == pytest.approx(x + fx)


def test_rslnk_zero_branch_is_the_identity():
    """The property the whole architecture rests on: a residual block can
    learn to do nothing, so adding depth cannot make the function class
    smaller."""
    rng = np.random.default_rng(2203)
    x = rng.standard_normal((4, 6))
    assert np.asarray(res(x, np.zeros_like(x))["y"]) == pytest.approx(x)


def test_rslnk_accepts_a_callable_branch():
    x = np.array([1.0, 2.0, 3.0])
    assert np.asarray(res(x, lambda v: v * 2)["y"]) == pytest.approx(x * 3)


def test_rslnk_echoes_the_branch_output():
    x = np.array([1.0, 2.0])
    fx = np.array([10.0, 20.0])
    assert np.asarray(res(x, fx)["Fx"]) == pytest.approx(fx)


def test_rslnk_is_additive_in_the_branch():
    rng = np.random.default_rng(2207)
    x = rng.standard_normal(8)
    a, b = rng.standard_normal(8), rng.standard_normal(8)
    lhs = np.asarray(res(x, a + b)["y"])
    rhs = np.asarray(res(x, a)["y"]) + b
    assert lhs == pytest.approx(rhs)


def test_rslnk_default_branch_is_the_identity_giving_two_x():
    """f=None documents F = identity, so y = 2x."""
    x = np.array([1.0, 2.0, 3.0])
    assert np.asarray(res(x)["y"]) == pytest.approx(2 * x)


def test_rslnk_accepts_a_precomputed_branch_as_well_as_a_callable():
    """Passing an array used to die on "'numpy.ndarray' object is not
    callable" several frames in -- a signature that appears four times in the
    audit's red list. Both forms must now agree."""
    x = np.array([1.0, 2.0, 3.0])
    fx = np.array([0.5, -1.0, 2.0])
    from_array = np.asarray(res(x, fx)["y"])
    from_callable = np.asarray(res(x, lambda v: fx)["y"])
    assert from_array == pytest.approx(from_callable)


def test_rslnk_rejects_a_shape_mismatch():
    """x + F(x) requires F to preserve shape; a mismatch is a real bug in the
    surrounding block, not something to broadcast away."""
    with pytest.raises((ValueError, TypeError)):
        res(np.zeros((3, 4)), np.zeros((3, 5)))
