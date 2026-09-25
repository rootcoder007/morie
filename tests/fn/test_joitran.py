"""Tests for joitran.joseph_itransformer."""

import math

import pytest

from morie.fn.joitran import joseph_itransformer


def _w(r, c, s):
    return [[math.sin(s + 0.7 * i + 1.3 * j) * 0.5 for j in range(c)] for i in range(r)]


def _net(x):
    T, D, F, Hh = len(x[0]), 3, 4, 2
    return joseph_itransformer(x, _w(D, T, 0.1), [0.1, -0.2, 0.05], _w(D, D, 0.2), _w(D, D, 0.3),
                               _w(D, D, 0.4), _w(F, D, 0.5), [0.0] * F, _w(D, F, 0.6), [0.0] * D,
                               _w(Hh, D, 0.7), [0.0, 0.1])


def test_joitran_basic():
    """Variates are tokens and carry no position, so permuting them permutes
    the forecasts (Liu et al. 2024, eqs. 1-2); attention rows sum to one."""
    x = [[1.0, 2.0, 0.5, -1.0], [0.3, -0.2, 0.8, 1.5], [2.0, 1.0, -0.5, 0.0]]
    a = _net(x)
    b = _net([x[2], x[0], x[1]])
    assert b["forecast"][0] == pytest.approx(a["forecast"][2], rel=1e-12)
    assert b["forecast"][1] == pytest.approx(a["forecast"][0], rel=1e-12)
    for row in a["attn"]:
        assert sum(row) == pytest.approx(1.0, rel=1e-14)
    assert (a["nvariates"], a["T"], a["D"], a["horizon"]) == (3, 4, 3, 2)


def test_joitran_edge():
    """One variate attends only to itself."""
    r = _net([[1.0, 2.0, 0.5, -1.0]])
    assert r["attn"] == [[1.0]] and r["attndiag"] == 1.0


