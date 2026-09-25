"""Tests for ksr023.kosorok_ch1_cox_estimating_equation."""

import math

import pytest

from morie.fn.ksr023 import kosorok_ch1_cox_estimating_equation


I = range(30)
Z = [((i * 7) % 11) / 5 - 1 for i in I]
T = [1 + ((i * 13) % 17) / 3 + 0.01 * i for i in I]
E = [0 if (i * 5) % 7 == 0 else 1 for i in I]


def test_ksr023_basic():
    """Eq. (1.4): U(beta) = sum over events of z_i - E_bar(t_i), the
    risk-set mean weighted by exp(beta z); recomputed here at beta = 0.3."""
    b = 0.3
    r = kosorok_ch1_cox_estimating_equation(b, Z, T, E)
    u = 0.0
    for i in I:
        if E[i]:
            risk = [j for j in I if T[j] >= T[i]]
            w = [math.exp(b * Z[j]) for j in risk]
            u += Z[i] - sum(wj * Z[j] for wj, j in zip(w, risk)) / sum(w)
    assert float(r["U_final"]) == pytest.approx(u / 30, rel=1e-12)   # (1/n) sum, eq. (1.4)


def test_ksr023_edge():
    """The root of U is the Cox estimate: at R survival::coxph's beta-hat
    (ties = breslow) the final score is zero."""
    r = kosorok_ch1_cox_estimating_equation(0.11502966611180812, Z, T, E)
    assert abs(float(r["U_final"])) < 1e-9
    # tied event times: each event counts once (Breslow)
    Tt = [round(t) for t in T]
    r = kosorok_ch1_cox_estimating_equation(0.3, Z, Tt, E)
    u = 0.0
    for i in I:
        if E[i]:
            risk = [j for j in I if Tt[j] >= Tt[i]]
            w = [math.exp(0.3 * Z[j]) for j in risk]
            u += Z[i] - sum(wj * Z[j] for wj, j in zip(w, risk)) / sum(w)
    assert float(r["U_final"]) == pytest.approx(u / 30, rel=1e-12)


