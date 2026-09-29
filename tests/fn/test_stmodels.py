"""Tests for stmodels: GTWR, multiscale GWR backfitting and Gaussian hidden Markov models."""

import itertools
import math

from morie.fn.stmodels import gaussian_hmm, gtwr_fit, mgwr_backfit

N = 30
XY = [((i * 7) % 10 + 0.3 * math.sin(i), (i * 3) % 8 + 0.2 * math.cos(i)) for i in range(N)]
TT = [i % 5 for i in range(N)]
X = [[math.sin(i * 0.7) + 0.1 * XY[i][0], math.cos(i * 1.1)] for i in range(N)]
Y = [1 + (0.5 + 0.1 * XY[i][0]) * X[i][0] - 0.3 * X[i][1] + 0.2 * math.sin(i * 2.3) for i in range(N)]


def _wls(w):
    Xr = [[1.0] + r for r in X]
    G = [[sum(w[j] * Xr[j][a] * Xr[j][b] for j in range(N)) for b in range(3)] for a in range(3)]
    h = [sum(w[j] * Xr[j][a] * Y[j] for j in range(N)) for a in range(3)]
    # Cramer-free: solve 3 x 3 by elimination
    M = [G[i] + [h[i]] for i in range(3)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][3] / M[i][i] for i in range(3)]


def test_gtwr_is_local_weighted_least_squares():
    r = gtwr_fit(Y, X, XY, TT, 6.0, lam=1.0, mu=0.5)
    i = 7
    w = []
    for j in range(N):
        d = math.sqrt((XY[i][0] - XY[j][0]) ** 2 + (XY[i][1] - XY[j][1]) ** 2 + 0.5 * (TT[i] - TT[j]) ** 2)
        w.append((1 - (d / 6.0) ** 2) ** 2 if d < 6.0 else 0.0)
    assert max(abs(a - b) for a, b in zip(r.beta[i], _wls(w))) <= 1e-10
    p = gtwr_fit(Y, X, XY, TT, 20.0, lam=0.4, distance="fotheringham", past_only=True)
    assert len(p.beta) == N


def test_mgwr_with_huge_bandwidths_is_ols():
    r = mgwr_backfit(Y, X, XY, [1e9, 1e9, 1e9], center=False)
    ols = _wls([1.0] * N)
    for b in r.beta:
        assert max(abs(a - c) for a, c in zip(b, ols)) <= 1e-8
    c = mgwr_backfit(Y, X, XY, [1e9, 1e9, 1e9])
    assert max(abs(a - b) for a, b in zip(c.beta[0], ols)) <= 1e-8


def _loglik(x, mu, sd, A, pi):
    K = len(mu)
    tot = 0.0
    for s in itertools.product(range(K), repeat=len(x)):
        p = pi[s[0]]
        for t in range(len(x)):
            if t:
                p *= A[s[t - 1]][s[t]]
            p *= math.exp(-0.5 * ((x[t] - mu[s[t]]) / sd[s[t]]) ** 2) / (sd[s[t]] * math.sqrt(2 * math.pi))
        tot += p
    return math.log(tot)


def test_hmm_loglik_matches_enumeration_and_viterbi():
    x = [0.1, 1.9, 2.2, -0.3, 0.4, 2.5, 2.1]
    r = gaussian_hmm(x, 2, max_iter=50)
    assert abs(r.loglik - _loglik(x, r.means, r.sds, r.trans, r.init)) <= 1e-9
    assert r.states[0] == [0, 1, 1, 0, 0, 1, 1]
    one = gaussian_hmm(x, 2, max_iter=1)
    assert r.loglik >= one.loglik - 1e-12
