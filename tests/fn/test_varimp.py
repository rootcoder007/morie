"""Tests for varimp.var_impulse_response (orthogonalised VAR IRF)."""

import math

import pytest

from morie.fn.varimp import var_impulse_response


A1 = [[0.5, 0.1], [0.3, 0.4]]
A2 = [[-0.2, 0.0], [0.0, 0.1]]
COEF = [[0.1] + A1[0] + A2[0], [-0.2] + A1[1] + A2[1]]
SIGMA = [[1.0, 0.3], [0.3, 0.5]]


def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def _irf(H, shock):
    """Phi_0 = I, Phi_h = A1 Phi_{h-1} + A2 Phi_{h-2}; Theta_h = Phi_h P
    with P the lower Cholesky factor of Sigma (Luetkepohl 2005, 2.3)."""
    P = [[math.sqrt(SIGMA[0][0]), 0.0], [0.0, 0.0]]
    P[1][0] = SIGMA[1][0] / P[0][0]
    P[1][1] = math.sqrt(SIGMA[1][1] - P[1][0] ** 2)
    Phi = [[[1.0, 0.0], [0.0, 1.0]]]
    for h in range(1, H + 1):
        t = _mm(A1, Phi[h - 1])
        if h >= 2:
            u = _mm(A2, Phi[h - 2])
            t = [[t[i][j] + u[i][j] for j in range(2)] for i in range(2)]
        Phi.append(t)
    return [[_mm(F, P)[i][shock] for i in range(2)] for F in Phi]


@pytest.mark.parametrize("shock", [0, 1])
def test_varimp_basic(shock):
    """Every response equals the MA recursion times the Cholesky factor
    (statsmodels VAR(...).irf().orth_irfs agrees to 1e-16 on a fitted
    VAR(2))."""
    r = var_impulse_response(COEF, SIGMA, horizon=8, shock_var=shock)
    irf = r.extra["irf"] if hasattr(r, "extra") else r["irf"]
    ref = _irf(8, shock)
    for h in range(9):
        assert [float(v) for v in irf[h]] == pytest.approx(ref[h], abs=1e-14)


def test_varimp_edge():
    """A 1-D coefficient vector and a non-conformable covariance raise."""
    with pytest.raises(ValueError):
        var_impulse_response([0.1, 0.5], SIGMA)
    with pytest.raises(ValueError):
        var_impulse_response(COEF, [[1.0]])


