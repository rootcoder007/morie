"""Tests for spiso: the isotropy condition (Schabenberger & Gotway 2005, Sec 2.2).

A second-order stationary random field is isotropic when the semivariogram
depends on the lag only through its length, gamma(h) = gamma(||h||). The module
screens for that by computing directional semivariograms and asking how far
apart they spread relative to their common level.
"""

import numpy as np
import pytest

from morie.fn.spiso import schabenberger_isotropy_condition as spiso


def _lattice(n=24, step=2.4):
    g = np.arange(n) / step
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _field(coords):
    return np.sin(coords[:, 0] * 0.8) + np.cos(coords[:, 1] * 0.8)


def test_isotropic_field_passes():
    coords = _lattice()
    res = spiso(coords, _field(coords))
    assert res["is_isotropic"]
    assert res["relative_spread"] < res["tol"]


def test_geometric_anisotropy_is_detected():
    """Stretching one axis makes the range direction-dependent -- the
    geometric anisotropy of Sec 2.2 -- so the screen must fail."""
    coords = _lattice()
    z = _field(coords)
    stretched = coords.copy()
    stretched[:, 0] *= 3.0
    res = spiso(stretched, z)
    assert not res["is_isotropic"]
    assert res["relative_spread"] > spiso(coords, z)["relative_spread"]


def test_result_does_not_depend_on_point_order():
    """A lag and its negation are the same direction, so relisting the points
    must not move a pair between sectors. Before the lags were oriented into
    one half-space this moved the answer in the third decimal: atan2 of a
    reversed pair folds back one ulp away, and on a regular lattice thousands
    of pairs sit exactly on a sector boundary."""
    coords = _lattice()
    z = _field(coords)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    a = spiso(coords, z)["relative_spread"]
    b = spiso(coords[order], z[order])["relative_spread"]
    assert a == pytest.approx(b, abs=1e-12)


def test_diagonal_lags_are_not_split_by_rounding():
    """Every 45-degree lag on this lattice lies exactly on the pi/4 sector
    boundary. Rescaling the coordinates changes the floating-point offsets but
    not the geometry, so the sector assignment must not move with them."""
    coords = _lattice()
    z = _field(coords)
    a = spiso(coords, z)["relative_spread"]
    b = spiso(coords * 4.0, z)["relative_spread"]
    assert a == pytest.approx(b, rel=1e-9)


def test_rejects_bad_input():
    coords = _lattice(n=4)
    with pytest.raises(ValueError):
        spiso(coords, np.ones(3))
    with pytest.raises(ValueError):
        spiso(np.ones((16, 3)), np.ones(16))
    with pytest.raises(ValueError):
        spiso(coords, _field(coords), n_dir=1)
