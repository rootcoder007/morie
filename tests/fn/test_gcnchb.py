"""Tests for gcnchb.chebnet."""

import pytest

from morie.fn import _array_core as np
from morie.fn.gcnchb import chebnet


def _laplacian(n, seed):
    rng = np.random.default_rng(seed)
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(rng.uniform(0, 1)) < 0.4:
                a[i][j] = a[j][i] = 1.0
    return [[(sum(a[i]) if i == j else 0.0) - a[i][j] for j in range(n)] for i in range(n)]


def test_gcnchb_basic():
    """The recurrence equals the spectral filter U g(Lambda) U' X, eq (4)."""
    n, p = 10, 5
    L = _laplacian(n, 42)
    X = np.random.default_rng(42).normal(0, 1, (n, p)).tolist()
    theta = [0.7, -0.4, 0.25, 0.1]
    r = chebnet(L, X, K=4, theta=theta)
    w, U = np.linalg.eigh(np.array(L))
    w = [float(t) for t in w]
    U = np.array(U).tolist()
    lmax = max(w)
    assert r["lambda_max"] == pytest.approx(lmax, rel=1e-12)
    g = []
    for lam in w:
        x = 2.0 * lam / lmax - 1.0
        t = [1.0, x]
        for _ in range(2, len(theta)):
            t.append(2.0 * x * t[-1] - t[-2])
        g.append(sum(c * tk for c, tk in zip(theta, t)))
    for i in range(n):
        for j in range(p):
            want = sum(U[i][a] * g[a] * U[m][a] * X[m][j] for a in range(n) for m in range(n))
            assert r["H"][i][j] == pytest.approx(want, abs=1e-12)


def test_gcnchb_edge():
    """K = 1 is theta_0 X; a non-square Laplacian is rejected."""
    L = _laplacian(6, 3)
    X = [[float(i + j) for j in range(2)] for i in range(6)]
    r = chebnet(L, X, K=1, theta=[2.5])
    assert r["H"] == [[2.5 * v for v in row] for row in X]
    with pytest.raises(ValueError, match="must be square"):
        chebnet([[1.0, 2.0, 3.0]], X)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gcnchb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
