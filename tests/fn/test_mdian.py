"""Tests for morie.fn.mdian: covariates residualised out before Baron-Kenny."""

import math

from morie.fn import _array_core as np
from morie.fn.mdian import mediation_analysis

X = [math.sin(k) for k in range(14)]
M = [0.5 * X[k] + 0.3 * math.cos(2 * k) for k in range(14)]
Y = [0.4 * X[k] + 0.8 * M[k] + 0.2 * math.sin(3 * k) for k in range(14)]


def _ols(cols, y):
    A = np.column_stack([np.ones(len(y))] + [np.array(c) for c in cols])
    return (np.linalg.inv(A.T @ A) @ (A.T @ np.array(y))).tolist()


def test_with_covariate_equals_full_regression():
    C = [math.cos(0.9 * k) for k in range(14)]
    r = mediation_analysis(Y, X, M, X=C)
    # Frisch-Waugh: the X and M coefficients in the regression that also contains C
    _, cp, b, _ = _ols([X, M, C], Y)
    _, a, _ = _ols([X, C], M)
    assert abs(r["c_prime"] - cp) < 1e-12 and abs(r["b"] - b) < 1e-12 and abs(r["a"] - a) < 1e-12
