"""Tests for spmrit.schabenberger_moran_i_residuals.

Reference values are spdep 1.3 lm.morantest(lm(y ~ x1 + x2), style "B").
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

from morie.fn.spmrit import schabenberger_moran_i_residuals


def test_spmrit_basic():
    """I, E and Var match lm.morantest when the raw response and x are given."""
    r = schabenberger_moran_i_residuals(Y, W, X)
    assert r["k"] == 3
    assert r["i"] == pytest.approx(0.12677553905685854, rel=1e-12)
    assert r["expectation"] == pytest.approx(-0.040517768878611776, rel=1e-12)
    assert r["variance"] == pytest.approx(0.03036576122101374, rel=1e-12)
    assert r["z"] == pytest.approx(
        (r["i"] - r["expectation"]) / math.sqrt(r["variance"]), rel=1e-12)


def test_spmrit_edge():
    """Passing the OLS residuals themselves gives the same answer (M is idempotent)."""
    from morie.fn._spx import solve
    xtx = [[sum(X[r][a] * X[r][b] for r in range(N)) for b in range(3)] for a in range(3)]
    xty = [sum(X[r][a] * Y[r] for r in range(N)) for a in range(3)]
    beta = solve(xtx, xty)
    e = [Y[r] - sum(X[r][a] * beta[a] for a in range(3)) for r in range(N)]
    r1 = schabenberger_moran_i_residuals(e, W, X)
    r2 = schabenberger_moran_i_residuals(Y, W, X)
    assert r1["i"] == pytest.approx(r2["i"], rel=1e-12)
    with pytest.raises(ValueError, match="at least 4 sites"):
        schabenberger_moran_i_residuals([1.0, 2.0, 3.0], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
