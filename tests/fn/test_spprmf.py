"""Tests for morie.fn.spprmf: the impacts recomputed in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.spprmf import spprmf

N = 10
W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in W]
X = [[1.0, v, math.cos(k)] for k, v in enumerate((2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1))]
B = [-0.2, 0.9, 0.4]
RHO = 0.35


def _ref(dens):
    Si = np.linalg.inv(np.eye(N) - RHO * np.array(W))
    eta = Si @ (np.array(X) @ np.array(B))
    d = [dens(float(v)) for v in eta.tolist()]
    dirr = sum(d[i] * float(Si[i, i]) for i in range(N)) / N
    tot = sum(d[i] * float(Si[i].sum()) for i in range(N)) / N
    return dirr, tot


def _phi(v):
    return math.exp(-v * v / 2) / math.sqrt(2 * math.pi)


def _lam(v):
    return math.exp(v) / (1 + math.exp(v)) ** 2


def test_lesage_pace_impacts():
    dirr, tot = _ref(_phi)
    r = spprmf(B, RHO, X, W)
    for k in range(2):
        assert abs(r["direct"][k] - dirr * B[k + 1]) < 1e-12
        assert abs(r["total"][k] - tot * B[k + 1]) < 1e-12
        assert abs(r["indirect"][k] - (tot - dirr) * B[k + 1]) < 1e-12


def test_marginal_method_matches_finite_difference():
    from morie.fn.spprprd import spprprd

    r = spprmf(B, RHO, X, W, method="marginal")
    h = 1e-6
    tot = 0.0
    for j in range(N):
        Xp = [row[:] for row in X]
        Xm = [row[:] for row in X]
        Xp[j][1] += h
        Xm[j][1] -= h
        pp, pm = spprprd(B, Xp, W, rho=RHO), spprprd(B, Xm, W, rho=RHO)
        tot += sum((a - b) / (2 * h) for a, b in zip(pp, pm))
    assert abs(r["total"][0] - tot / N) < 1e-7
