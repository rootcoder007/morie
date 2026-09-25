"""Tests for spcnds.schabenberger_conditional_sim (eq. 7.1)."""

import math

import pytest

from morie.fn._schab_sim import simulate_unconditional
from morie.fn.spcnds import schabenberger_conditional_sim


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _cov(n=6):
    s = [0.0, 0.7, 1.9, 2.4, 3.3, 4.1][:n]
    return [[math.exp(-abs(a - b) / 1.5) for b in s] for a in s]


@pytest.mark.parametrize("method", ["cholesky", "spectral"])
def test_spcnds_basic(method):
    """Zc = mu + S + c' Sigma_obs^-1 ((z - mu) - S_obs) with S the
    centred unconditional draw of the same stream; the realisation
    reproduces the data at the sampled sites, and the simple-kriging
    variance is C(s,s) - c' Sigma_obs^-1 c."""
    C = _cov()
    z = [1.2, 0.4, -0.3]
    mu = 0.5
    S = [float(v) for v in simulate_unconditional([0.0] * 6, C, method=method, seed=3, stream=1)]
    r = schabenberger_conditional_sim(C, z, 3, mean=mu, method=method, seed=3, stream=1)
    Sig = [row[:3] for row in C[:3]]
    w = _solve(Sig, [(z[i] - mu) - S[i] for i in range(3)])
    exp = [mu + S[k] + sum(C[k][j] * w[j] for j in range(3)) for k in range(6)]
    assert [float(v) for v in r["field"]] == pytest.approx(exp, abs=1e-12)
    assert [float(v) for v in r["field"]][:3] == pytest.approx(z, abs=1e-12)
    assert r["honors_data"] < 1e-12
    for k in range(6):
        q = _solve(Sig, [C[k][j] for j in range(3)])
        sk = C[k][k] - sum(C[k][j] * q[j] for j in range(3))
        assert float(r["kriging_variance"][k]) == pytest.approx(sk, abs=1e-12)


def test_spcnds_edge():
    """n_obs must leave a target, z must have n_obs entries, and the
    method must be one of the two roots."""
    C = _cov()
    with pytest.raises(ValueError):
        schabenberger_conditional_sim(C, [0.0] * 6, 6)
    with pytest.raises(ValueError):
        schabenberger_conditional_sim(C, [0.0] * 2, 3)
    with pytest.raises(ValueError):
        schabenberger_conditional_sim(C, [0.0] * 3, 3, method="exponential")
