"""Tests for netcms.network_psychometrics (graphical lasso, ESL Alg. 17.2)."""

import math

import pytest

from morie.fn.netcms import network_psychometrics


DATA = [[math.sin(1.1 * i), math.sin(1.1 * i) + 0.5 * math.cos(2.3 * i), math.cos(0.7 * i),
         0.4 * math.cos(0.7 * i) + math.sin(3.1 * i)] for i in range(60)]


def _cov(rows):
    n, p = len(rows), len(rows[0])
    mu = [sum(r[j] for r in rows) / n for j in range(p)]
    return [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in rows) / n for b in range(p)] for a in range(p)]


def test_netcms_basic():
    """Optimality of the penalised likelihood (ESL eq. 17.22): W = Theta^-1,
    diag W = diag S + lambda, |W - S| <= lambda off the diagonal with
    equality lambda sign(theta_jk) wherever theta_jk != 0; partial
    correlations are -theta_jk / sqrt(theta_jj theta_kk)."""
    lam = 0.05
    r = network_psychometrics(DATA, lam=lam, tol=1e-12, maxit=2000)
    S = _cov(DATA)
    Th, W = r["precision"], r["covariance_fit"]
    p = 4
    for a in range(p):
        for b in range(p):
            wt = sum(W[a][k] * Th[k][b] for k in range(p))
            assert wt == pytest.approx(1.0 if a == b else 0.0, abs=1e-8)
            if a == b:
                assert W[a][a] == pytest.approx(S[a][a] + lam, rel=1e-12)
            elif abs(Th[a][b]) > 1e-10:
                assert W[a][b] - S[a][b] == pytest.approx(lam * math.copysign(1, Th[a][b]), abs=1e-8)
            else:
                assert abs(W[a][b] - S[a][b]) <= lam + 1e-9
            if a != b:
                assert r["partial_correlations"][a][b] == pytest.approx(-Th[a][b] / math.sqrt(Th[a][a] * Th[b][b]), rel=1e-12)
    assert r["n_edges"] == sum(r["adjacency"][a][b] for a in range(p) for b in range(a + 1, p))
    # lambda = 0 is the unpenalised MLE: Theta = S^-1, so W = S
    z = network_psychometrics(S=S, lam=0.0, tol=1e-13, maxit=5000)
    assert z["covariance_fit"] == [pytest.approx(row, abs=1e-8) for row in S]


def test_netcms_edge():
    """Negative penalties and missing inputs raise."""
    with pytest.raises(ValueError):
        network_psychometrics(DATA, lam=-0.1)
    with pytest.raises(ValueError):
        network_psychometrics()
