"""Tests for gebvs.gebv_selection (VanRaden G, GBLUP, truncation selection)."""

from statistics import NormalDist

import pytest

from morie.fn.gebvs import gebv_selection

M = [[0, 1, 2, 1], [2, 1, 0, 0], [1, 1, 1, 2], [0, 0, 2, 1], [2, 2, 1, 0], [1, 0, 0, 1]]
Y = [4.1, 2.0, 3.3, 5.2, 1.7, 2.9]


def _vanraden():
    n, m = len(M), len(M[0])
    p = [sum(r[j] for r in M) / (2 * n) for j in range(m)]
    Z = [[r[j] - 2 * p[j] for j in range(m)] for r in M]
    den = 2 * sum(q * (1 - q) for q in p)
    return Z, [[sum(a * b for a, b in zip(Z[i], Z[k])) / den for k in range(n)] for i in range(n)]


def _solve(A, b):
    n = len(b)
    T = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(T[r][c]))
        T[c], T[piv] = T[piv], T[c]
        for r in range(n):
            if r != c:
                f = T[r][c] / T[c][c]
                T[r] = [a - f * v for a, v in zip(T[r], T[c])]
    return [T[i][n] / T[i][i] for i in range(n)]


def test_gebvs_basic():
    """GBLUP g = G (G + lam I)^-1 (y - ybar), lam = (1 - h2) / h2, on
    VanRaden's G recomputed; ranking is descending g and the selection
    intensity for the top 2 of 6 is phi(z) / (1/3), z = Phi^-1(2/3)."""
    _, G = _vanraden()
    h2 = 0.4
    lam = (1 - h2) / h2
    ybar = sum(Y) / len(Y)
    A = [[G[i][k] + (lam if i == k else 0.0) for k in range(6)] for i in range(6)]
    a = _solve(A, [v - ybar for v in Y])
    g = [sum(G[i][k] * a[k] for k in range(6)) for i in range(6)]
    result = gebv_selection(M, y=Y, h2=h2, n_select=2)
    assert isinstance(result, dict)
    assert [float(v) for v in result["gebv"]] == pytest.approx(g, rel=1e-10, abs=1e-12)
    assert [int(v) for v in result["ranking"]] == sorted(range(6), key=lambda i: -g[i])
    z = NormalDist().inv_cdf(2 / 3)
    assert result["selection_intensity"] == pytest.approx(NormalDist().pdf(z) * 3, rel=1e-9)


def test_gebvs_edge():
    """Known marker effects give g = Z beta directly; a genotype outside
    {0, 1, 2} is refused."""
    Z, _ = _vanraden()
    beta = [0.5, -1.0, 0.25, 2.0]
    r = gebv_selection(M, effects=beta)
    assert [float(v) for v in r["gebv"]] == pytest.approx(
        [sum(z * b for z, b in zip(row, beta)) for row in Z], rel=1e-12, abs=1e-14)
    with pytest.raises(ValueError):
        gebv_selection([[0, 3], [1, 1]], effects=[1.0, 1.0])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gebvs as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
