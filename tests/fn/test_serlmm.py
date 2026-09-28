"""serlmm: the profile likelihood equals the Gaussian density at the GLS estimates; the fit is a maximum."""

import math

import pytest

from morie.fn._qpcore import inverse, solve
from morie.fn._rng import random_normal
from morie.fn.serlmm import lmm_serial_covariance, lmm_serial_fit, lmm_serial_loglik, serial_correlation

_e = [float(v) for v in random_normal(120, seed=17)]
NS, NT = 8, 4
SID = [s for s in range(NS) for _ in range(NT)]
TT = [[0.0, 1.0, 3.0, 4.0][j] + 0.2 * _e[s] for s in range(NS) for j in range(NT)]
XX = [[1.0, TT[i], _e[10 + i]] for i in range(NS * NT)]
YY = [0.5 + 0.3 * TT[i] + XX[i][2] + 1.2 * _e[50 + SID[i]] + 0.7 * _e[60 + i] for i in range(NS * NT)]


def _logdet(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(v) if i == j else v / L[j][j]
    return 2 * sum(math.log(L[i][i]) for i in range(n))


def _direct(sb2, t2, phi, n2, beta, s2, reml):
    tot = t2 + n2
    ll, xvx = 0.0, [[0.0] * 3 for _ in range(3)]
    for s in range(NS):
        rows = [i for i in range(NS * NT) if SID[i] == s]
        V = lmm_serial_covariance([TT[i] for i in rows], sb2 / tot * s2, t2 / tot * s2, phi, n2 / tot * s2)
        r = [YY[i] - sum(XX[i][a] * beta[a] for a in range(3)) for i in rows]
        ll += -0.5 * (len(rows) * math.log(2 * math.pi) + _logdet(V) + sum(a * b for a, b in zip(r, solve(V, r))))
        Vi = inverse(V)
        for a in range(3):
            for b in range(3):
                xvx[a][b] += sum(XX[rows[j]][a] * Vi[j][k] * XX[rows[k]][b] for j in range(NT) for k in range(NT))
    if reml:
        ll += 1.5 * math.log(2 * math.pi) - 0.5 * _logdet(xvx)
    return ll


def test_serial_correlation_forms():
    assert serial_correlation([0.0, 2.0], 0.3) == pytest.approx([1.0, math.exp(-0.6)], rel=1e-15)
    assert serial_correlation([1.5], 0.3, "gaussian") == pytest.approx([math.exp(-0.675)], rel=1e-15)
    assert serial_correlation([3.0], 0.5, "ar1") == pytest.approx([0.125], rel=1e-15)
    with pytest.raises(ValueError):
        serial_correlation([1.0], 0.5, "linear")


def test_profile_loglik_matches_direct_density():
    for reml in (False, True):
        r = lmm_serial_loglik(YY, XX, SID, TT, 0.6, 0.9, 0.4, 0.3, reml=reml)
        assert r.loglik == pytest.approx(_direct(0.6, 0.9, 0.4, 0.3, r.beta, r.sigma2, reml), rel=1e-11)
        worse = [b + 0.01 for b in r.beta]
        assert _direct(0.6, 0.9, 0.4, 0.3, worse, r.sigma2, False) < _direct(
            0.6, 0.9, 0.4, 0.3, r.beta, r.sigma2, False
        )


def test_fit_is_a_local_maximum():
    f = lmm_serial_fit(YY, XX, SID, TT)
    assert f.converged
    base = lmm_serial_loglik(YY, XX, SID, TT, f.sigma_b2, f.tau2, f.phi, f.nu2).loglik
    assert base == pytest.approx(f.loglik, rel=1e-12)
    for d in ((1.05, 1, 1), (0.95, 1, 1), (1, 1.05, 1), (1, 0.95, 1)):
        other = lmm_serial_loglik(YY, XX, SID, TT, f.sigma_b2 * d[0] + 1e-3 * (d[0] - 1), f.tau2, f.phi * d[1], f.nu2)
        assert other.loglik <= base + 1e-9
