"""Tests for morie.fn.medfm: NDE and NIE recomputed from cell means."""

from morie.fn.medfm import mediation_formula

x = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
m = [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
y = [0.5, 1.0, 1.5, 0.7, 2.0, 2.2, 1.1, 2.5, 0.9, 2.4, 1.9, 1.2]


def _e(xv, mv):
    c = [y[i] for i in range(12) if x[i] == xv and m[i] == mv]
    return sum(c) / len(c)


def _p(mv, xv):
    return sum(1 for i in range(12) if x[i] == xv and m[i] == mv) / sum(1 for v in x if v == xv)


def test_nde_nie():
    r = mediation_formula(x, m, y, x1=1, x0=0)
    nde = sum((_e(1, mv) - _e(0, mv)) * _p(mv, 0) for mv in (0, 1))
    nie = sum(_e(1, mv) * (_p(mv, 1) - _p(mv, 0)) for mv in (0, 1))
    assert abs(r["nde"] - nde) < 1e-14 and abs(r["nie"] - nie) < 1e-14
    assert abs(r["te"] - (nde + nie)) < 1e-14
