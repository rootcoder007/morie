"""Tests for cauchw.cauchy_weight (Holland & Welsch 1977 Cauchy weight)."""

import pytest

from morie.fn.cauchw import cauchy_weight

C = 2.3849


def test_closed_form_values():
    r = cauchy_weight([0.0, C, -C], C)
    assert r["weights"][0] == 1.0
    assert r["weights"][1] == 0.5
    assert r["weights"][2] == 0.5
    assert r["rho"][0] == 0.0
    assert r["psi"][0] == 0.0


def test_rho_derivative_is_psi():
    h = 1e-6
    rs = [0.3, 1.0, C, 5.0, 20.0]
    up = cauchy_weight([v + h for v in rs], C)["rho"]
    dn = cauchy_weight([v - h for v in rs], C)["rho"]
    psi = cauchy_weight(rs, C)["psi"]
    for i in range(len(rs)):
        assert abs((up[i] - dn[i]) / (2 * h) - psi[i]) < 1e-7


def test_psi_peaks_at_c_and_redescends():
    grid = [0.01 * i for i in range(1, 1201)]
    psi = cauchy_weight(grid, C)["psi"]
    top = grid[psi.index(max(psi))]
    assert abs(top - C) < 0.02
    assert cauchy_weight([2 * C], C)["psi"][0] < cauchy_weight([C], C)["psi"][0]


def test_weights_decrease_but_never_reach_zero():
    grid = [0.1 * i for i in range(201)]
    w = cauchy_weight(grid, C)["weights"]
    assert all(w[i + 1] < w[i] for i in range(len(w) - 1))
    assert cauchy_weight([1e8], C)["weights"][0] > 0.0


def test_error_paths():
    with pytest.raises(ValueError):
        cauchy_weight([], 2.0)
    with pytest.raises(ValueError):
        cauchy_weight([1.0], 0.0)
    with pytest.raises(ValueError):
        cauchy_weight([1.0], -1.0)
