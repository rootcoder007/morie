"""Tests for spblk.spatial_block_kriging."""

import numpy as np
import pytest

from morie.fn.okrig import ordinary_kriging
from morie.fn.spblk import spatial_block_kriging


def test_spblk_degenerate_block_equals_point_kriging():
    """A block collapsed to a single quadrature point IS point kriging, so
    the two implementations must agree on it."""
    x = [1.0, 3.0, 2.0, 5.0]
    coords = [[0.0], [1.0], [2.0], [3.0]]
    blk = spatial_block_kriging(x, coords, [np.array([[1.5]])])
    pt = ordinary_kriging(x, coords, [[1.5]])
    assert float(blk["estimate"][0]) == pytest.approx(float(pt["estimate"]), rel=1e-9)


def test_spblk_constant_field_is_reproduced_with_lower_variance():
    """Unbiasedness reproduces a constant field exactly, and averaging over
    a block cannot be harder than predicting a point: the block SE is at
    most the point SE at the block centre."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 3, size=(12, 2))
    const = spatial_block_kriging([4.0] * 12, coords, [(np.array([1.0, 1.0]), np.array([2.0, 2.0]))])
    assert float(const["estimate"][0]) == pytest.approx(4.0, abs=1e-8)

    x = rng.normal(size=12)
    blk = spatial_block_kriging(x, coords, [(np.array([1.0, 1.0]), np.array([2.0, 2.0]))], n_quad=16)
    pt = ordinary_kriging(x, coords, [[1.5, 1.5]])
    assert float(blk["se"][0]) <= float(pt["se"]) + 1e-9


def test_spblk_several_blocks_come_back_in_order():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    coords = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    r = spatial_block_kriging(x, coords, [np.array([[0.5]]), np.array([[3.5]])])
    a, b = (float(v) for v in r["estimate"])
    # The field increases along the line, so the left block must predict less.
    assert a < b


def test_spblk_rejects_bad_input():
    with pytest.raises(ValueError, match="coords rows"):
        spatial_block_kriging([1.0, 2.0], [[0.0], [1.0], [2.0]], [np.array([[0.5]])])
    with pytest.raises(ValueError, match="d<=2"):
        coords3 = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
        spatial_block_kriging([1.0, 2.0, 3.0], coords3, [(np.zeros(3), np.ones(3))])
