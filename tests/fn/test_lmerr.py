"""Tests for morie.fn.lmerr: recompute the statistic in matrix form."""

import math

from morie.fn import _array_core as np
from morie.fn.lmerr import lmerr

N = 14
W = [[0.0] * N for _ in range(N)]
for i in range(N):
    for d in (1, 2):
        W[i][(i + d) % N] = W[i][(i - d) % N] = 0.25
X = [[math.sin(i * 1.3) + i / 7.0, ((i * 5) % 7) / 3.0] for i in range(N)]
Y = [1.0 + 2.0 * X[i][0] - X[i][1] + ((i * 3) % 5 - 2) / 3.0 + 0.4 * math.cos(i) for i in range(N)]


def _ref():
    """Matrix-form recomputation (Anselin 1988; Anselin et al. 1996; Koley and Bera 2024)."""
    Wa = np.array(W)
    Xa = np.column_stack([np.ones(N), np.array([r[0] for r in X]), np.array([r[1] for r in X])])
    ya = np.array(Y)
    M = np.eye(N) - Xa @ np.linalg.inv(Xa.T @ Xa) @ Xa.T
    u = M @ ya
    s2 = float(u @ u) / N
    T = float(np.trace(Wa.T @ Wa + Wa @ Wa))
    wxb = Wa @ (ya - u)
    nJ = (float(wxb @ (M @ wxb)) + T * s2) / s2
    de = float(u @ (Wa @ u)) / s2
    dl = float(u @ (Wa @ ya)) / s2
    out = {"RSerr": de**2 / T, "RSlag": dl**2 / nJ, "adjRSlag": (dl - de) ** 2 / (nJ - T)}
    out["z"] = (de - T * dl / nJ) / math.sqrt(T * (1 - T / nJ))
    out["adjRSerr"] = out["z"] ** 2
    out["SARMA"] = out["adjRSlag"] + out["RSerr"]
    WX = Wa @ Xa[:, 1:]
    g = WX.T @ u
    out["RSWX"] = float(g @ np.linalg.inv(WX.T @ M @ WX) @ g) / s2
    G = np.column_stack([wxb, WX])
    J = G.T @ M @ G
    J[0, 0] += T * s2
    dd = np.array([dl] + [float(v) / s2 for v in g])
    out["joint"] = float(dd @ np.linalg.inv(J) @ dd) * s2
    return out


def _sf(x, df):
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def test_rserr():
    r = lmerr(Y, X, W)
    ref = _ref()["RSerr"]
    assert abs(r.statistic - ref) < 1e-10 * max(1.0, ref)
    assert abs(r.p_value - _sf(ref, 1)) < 1e-10
