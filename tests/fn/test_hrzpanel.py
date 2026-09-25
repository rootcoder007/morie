"""Tests for hrzpanel.horowitz_panel_deconv."""

import math
import random

import pytest

from morie.fn.hrzpanel import horowitz_panel_deconv


def _panel(n=1000, seed=3):
    rnd = random.Random(seed)
    U = [rnd.gauss(0, 1) for _ in range(n)]
    Y = [[u + rnd.gauss(0, 0.5), u + rnd.gauss(0, 0.5)] for u in U]
    X = [[[0.0], [0.0]] for _ in range(n)]
    return Y, X


def test_hrzpanel_basic():
    """With U ~ N(0, 1), eps ~ N(0, 0.25) and T = 2, the deconvolved f_U is
    the standard normal density to within 0.02 on the grid."""
    Y, X = _panel()
    g = [-2.0, 0.0, 1.0]
    r = horowitz_panel_deconv(Y, X, [0.0], grid_u=g, grid_z=[-0.5, 0.0, 0.5])
    for got, x in zip(r["f_U"].tolist(), g):
        assert abs(got - math.exp(-x * x / 2) / math.sqrt(2 * math.pi)) < 0.02
    assert (r["n"], r["T"], r["d"]) == (1000, 2, 1)


def test_hrzpanel_edge():
    """f_eps is estimated from the differences through the square-root
    identity, which assumes symmetry: it peaks at 0 and is symmetric."""
    Y, X = _panel()
    r = horowitz_panel_deconv(Y, X, [0.0], grid_u=[0.0], grid_z=[-0.5, 0.0, 0.5])
    lo, mid, hi = r["f_eps"].tolist()
    assert mid > lo and mid > hi and abs(lo - hi) < 0.02
    with pytest.raises(ValueError, match="10 individuals"):
        horowitz_panel_deconv(Y[:5], X[:5], [0.0])


