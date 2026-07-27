"""Tests for rotrp.rotary_position_embedding."""

import numpy as np
import pytest

from morie.fn.rotrp import rotary_position_embedding


def _rope(x, base=10000.0):
    return np.asarray(rotary_position_embedding(x, base=base)["y"], dtype=float)


def test_rotrp_is_norm_preserving():
    """RoPE rotates each coordinate pair, so it cannot change any row's norm."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 8))
    y = _rope(x)
    assert y.shape == x.shape
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), atol=1e-10)


def test_rotrp_inner_product_depends_only_on_relative_position():
    """The defining property of RoPE (Su et al. 2021): <f(q,m), f(k,n)> is a
    function of m - n alone. A stub that merely scales or shifts fails this."""
    rng = np.random.default_rng(1)
    q = rng.normal(size=8)
    k = rng.normal(size=8)

    def dot_at(m, n):
        rows = np.zeros((max(m, n) + 1, 8))
        rows[m] = q
        a = _rope(rows)[m]
        rows2 = np.zeros((max(m, n) + 1, 8))
        rows2[n] = k
        b = _rope(rows2)[n]
        return float(a @ b)

    # Same offset (m - n = 2) at two different absolute positions.
    assert dot_at(3, 1) == pytest.approx(dot_at(6, 4), abs=1e-9)
    # A different offset gives a genuinely different value, so the test above
    # is not passing merely because everything is equal.
    assert abs(dot_at(3, 1) - dot_at(5, 1)) > 1e-6


def test_rotrp_position_zero_is_the_identity():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 6))
    np.testing.assert_allclose(_rope(x)[0], x[0], atol=1e-12)


def test_rotrp_requires_even_d_model():
    with pytest.raises(ValueError, match="even"):
        rotary_position_embedding(np.zeros((4, 7)))
