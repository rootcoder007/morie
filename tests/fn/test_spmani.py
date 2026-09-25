"""Tests for spmani.schabenberger_mantel_standard.

The Gaussian moments E = s2 tr(AM), Var = 2 s2^2 tr(AMAM) were checked by
Monte Carlo (400,000 draws on this grid: mean -9.33 vs -9.3, variance
925.0 vs 925.4 at sigma^2 = 3).
"""

import math

import pytest

NR, NC = 5, 4
N = NR * NC
# rook adjacency on a 5 x 4 grid, sites numbered row by row
W = [[1.0 if abs(i // NC - j // NC) + abs(i % NC - j % NC) == 1 else 0.0
      for j in range(N)] for i in range(N)]
Z = [((i * 37) % 17) / 3 + 0.5 * (i // NC) for i in range(N)]
X1 = [float((i * 5) % 9) for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
Y = [1 + 0.8 * a - 0.5 * b + ((i * 13) % 7 - 3) / 2 + 0.3 * (i // NC)
     for i, (a, b) in enumerate(zip(X1, X2))]
X = [[1.0, a, b] for a, b in zip(X1, X2)]

from morie.fn.spmani import schabenberger_mantel_standard


def test_spmani_basic():
    """M2 = d'Wd with the Gaussian moments recomputed in the test."""
    r = schabenberger_mantel_standard(None, Z, W)
    zbar = sum(Z) / N
    d = [z - zbar for z in Z]
    s2 = sum(t * t for t in d) / (N - 1)
    m2 = sum(W[i][j] * d[i] * d[j] for i in range(N) for j in range(N))
    assert r["m2"] == pytest.approx(m2, rel=1e-12)
    # tr(A) = 0, so tr(AM) = -w../n
    s0 = sum(map(sum, W))
    assert r["expectation"] == pytest.approx(-s2 * s0 / N, rel=1e-12)
    m = [[(1.0 if i == j else 0.0) - 1.0 / N for j in range(N)] for i in range(N)]
    am = [[sum(W[i][k] * m[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
    tr = sum(am[i][k] * am[k][i] for i in range(N) for k in range(N))
    assert r["variance"] == pytest.approx(2.0 * s2 * s2 * tr, rel=1e-12)
    assert r["z"] == pytest.approx((m2 - r["expectation"]) / math.sqrt(r["variance"]), rel=1e-12)


def test_spmani_edge():
    """A supplied u is used verbatim and switches the Z-test off."""
    u = [[float(i * j % 3) for j in range(N)] for i in range(N)]
    r = schabenberger_mantel_standard(None, Z, W, u=u)
    assert r["m2"] == pytest.approx(
        sum(W[i][j] * u[i][j] for i in range(N) for j in range(N)), rel=1e-15)
    assert r["z"] is None and r["p_value"] is None
