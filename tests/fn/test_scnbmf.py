"""Tests for morie.fn.scnbmf: NB2 impacts equal the Poisson impacts (same conditional mean)."""

import math

from morie.fn.scnbmf import scnbmf, scnbmf_fn
from morie.fn.scpmf import scpmf

W = [[0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0]]
X = [[1.0, 0.2], [1.0, -0.4], [1.0, 0.9], [1.0, 0.3]]


def test_equals_poisson_impacts():
    r, p = scnbmf([0.1, 0.7], -0.25, X, W), scpmf([0.1, 0.7], -0.25, X, W)
    assert r.direct == p.direct and r.total == p.total and r.indirect == p.indirect
    assert scnbmf_fn is scnbmf


def test_no_spatial_lag():
    r = scnbmf([0.1, 0.7], 0.0, X, W)
    mbar = sum(math.exp(0.1 + 0.7 * x[1]) for x in X) / 4
    assert abs(r.direct[1] - 0.7 * mbar) < 1e-12 and abs(r.total[1] - 0.7 * mbar) < 1e-12
