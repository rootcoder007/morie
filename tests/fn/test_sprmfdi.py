"""Tests for morie.fn.sprmfdi: the impacts recomputed in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.sprmfdi import sprmfdi

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


def test_direct_indirect():
    dirr, tot = _ref(_phi)
    r = sprmfdi(B, RHO, X, W)
    assert abs(r["direct"][1] - dirr * B[2]) < 1e-12
    assert abs(r["indirect"][1] - (tot - dirr) * B[2]) < 1e-12
