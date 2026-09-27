"""Moran eigenvector spatial filtering (Tiefelsdorf and Griffith 2007).

Checked against spatialreg 1.3 SpatialFiltering: same eigenvector sequence,
Moran's I, z and R2 to 3e-13, fitted values to 1e-14 on a 40-point kNN graph;
tests/cross/test-morie_vs_spatialreg.R repeats that in R.
"""

import pytest

from morie.fn.mesf import moran_eigenvector_filter

N = 12
A = [[1 if abs(i - j) == 1 else 0 for j in range(N)] for i in range(N)]
Y = [1.0, 1.4, 2.2, 2.9, 3.1, 2.6, 2.0, 1.1, 0.8, 1.5, 2.4, 2.7]
X = [[1.0, v] for v in (0.2, 0.5, 0.1, 0.9, 0.4, 0.3, 0.8, 0.6, 0.7, 0.05, 0.35, 0.55)]


def _solve(M, b):
    n = len(b)
    T = [list(M[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(T[r][c]))
        T[c], T[p] = T[p], T[c]
        for r in range(n):
            if r != c:
                f = T[r][c] / T[c][c]
                T[r] = [a - f * b for a, b in zip(T[r], T[c])]
    return [T[i][n] / T[i][i] for i in range(n)]


def _fit(Z, y):
    q = len(Z[0])
    b = _solve(
        [[sum(r[a] * r[c] for r in Z) for c in range(q)] for a in range(q)],
        [sum(r[a] * t for r, t in zip(Z, y)) for a in range(q)],
    )
    return [sum(r[k] * b[k] for k in range(q)) for r in Z]


def _S():
    tot = sum(sum(r) for r in A)
    return [[N / tot * v for v in r] for r in A]


def test_documented_selection():
    r = moran_eigenvector_filter(Y, X, A)
    assert [row["evec"] for row in r.selection] == [0, 1, 3]
    assert r.stop_reason == "inversion"


def test_fitted_is_ols_on_design_plus_selected_vectors():
    r = moran_eigenvector_filter(Y, X, A)
    Z = [X[i] + list(r.vectors[i]) for i in range(N)]
    assert r.fitted == pytest.approx(_fit(Z, Y), abs=1e-10)


def test_selected_vectors_are_orthogonal_to_x():
    r = moran_eigenvector_filter(Y, X, A)
    for k in range(len(r.vectors[0])):
        v = [row[k] for row in r.vectors]
        for c in range(2):
            assert abs(sum(X[i][c] * v[i] for i in range(N))) < 1e-10


def test_step0_moran_is_the_ols_residual_moran():
    r = moran_eigenvector_filter(Y, X, A)
    f = _fit(X, Y)
    e = [Y[i] - f[i] for i in range(N)]
    S = _S()
    mi = sum(e[i] * S[i][j] * e[j] for i in range(N) for j in range(N)) / sum(v * v for v in e)
    assert r.selection[0]["moran"] == pytest.approx(mi, abs=1e-12)
    assert r.selection[0]["r2"] == pytest.approx(
        1 - sum(v * v for v in e) / sum((v - sum(Y) / N) ** 2 for v in Y), abs=1e-12
    )


def test_candidates_can_run_out():
    a10 = [row[:10] for row in A[:10]]
    r = moran_eigenvector_filter(Y[:10], X[:10], a10)
    assert r.stop_reason == "exhausted"
    assert len(r.selection) == 5
