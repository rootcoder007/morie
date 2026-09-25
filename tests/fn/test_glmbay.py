"""Tests for glmbay.bayesian_glm (penalised IRLS + Laplace posterior)."""

import math

import pytest

from morie.fn.glmbay import bayesian_glm

X = [[0.5], [-1.0], [2.0], [0.0], [1.5], [-0.5]]
Y = [1.2, -0.4, 2.9, 0.3, 2.1, 0.0]
PS = 1.5


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b2 for a, b2 in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def _logdet(A):
    n = len(A)
    M = [row[:] for row in A]
    s = 0.0
    for c in range(n):
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        s += math.log(M[c][c])
    return s


def test_glmbay_basic():
    """Gaussian family (unit noise): the mode is (X'X + I/s^2)^-1 X'y, the
    covariance its inverse, and the Laplace log marginal is exact:
    y ~ N(0, I + s^2 X X')."""
    result = bayesian_glm(X, Y, family="gaussian", prior_sd=PS)
    assert isinstance(result, dict)
    Z = [[1.0] + r for r in X]
    tau = 1.0 / PS ** 2
    A = [[sum(z[a] * z[b] for z in Z) + (tau if a == b else 0.0) for b in range(2)]
         for a in range(2)]
    beta = _solve(A, [sum(z[a] * y for z, y in zip(Z, Y)) for a in range(2)])
    assert result["coefficients"] == pytest.approx(beta, rel=1e-10, abs=1e-12)
    cov00 = _solve(A, [1.0, 0.0])[0]
    assert result["posterior_sd"][0] == pytest.approx(math.sqrt(cov00), rel=1e-10)
    n = len(Y)
    S = [[(1.0 if i == j else 0.0) + PS ** 2 * sum(a * b for a, b in zip(Z[i], Z[j]))
          for j in range(n)] for i in range(n)]
    Sy = _solve(S, Y)
    lm = -0.5 * (n * math.log(2 * math.pi) + _logdet(S) + sum(a * b for a, b in zip(Y, Sy)))
    assert result["log_marginal"] == pytest.approx(lm, rel=1e-10)


def test_glmbay_edge():
    """Completely separated binary data: ML would diverge, the posterior
    mode is finite and satisfies X'(y - mu) = beta / s^2."""
    yb = [1.0 if r[0] > 0.25 else 0.0 for r in X]
    r = bayesian_glm(X, yb, family="binomial", prior_sd=PS)
    assert r["converged"]
    b = r["coefficients"]
    assert all(math.isfinite(v) for v in b)
    Z = [[1.0] + x for x in X]
    mu = [1.0 / (1.0 + math.exp(-(b[0] * z[0] + b[1] * z[1]))) for z in Z]
    for a in range(2):
        g = sum(z[a] * (y - m) for z, y, m in zip(Z, yb, mu)) - b[a] / PS ** 2
        assert abs(g) < 1e-8
    with pytest.raises(ValueError):
        bayesian_glm(X, Y[:3])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.glmbay as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
