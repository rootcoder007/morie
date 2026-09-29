"""Tests for morie.fn.causmedb: the three Baron-Kenny regressions recomputed."""

import math

from morie.fn import _array_core as np
from morie.fn.causmedb import causal_mediation_baron_kenny

X = [math.sin(k) for k in range(14)]
M = [0.5 * X[k] + 0.3 * math.cos(2 * k) for k in range(14)]
Y = [0.4 * X[k] + 0.8 * M[k] + 0.2 * math.sin(3 * k) for k in range(14)]


def _ols(cols, y):
    A = np.column_stack([np.ones(len(y))] + [np.array(c) for c in cols])
    return (np.linalg.inv(A.T @ A) @ (A.T @ np.array(y))).tolist()


def test_paths():
    r = causal_mediation_baron_kenny(X, M, Y)
    c = _ols([X], Y)[1]
    a = _ols([X], M)[1]
    _, cp, b = _ols([X, M], Y)
    assert abs(r["c"] - c) < 1e-12 and abs(r["a"] - a) < 1e-12
    assert abs(r["b"] - b) < 1e-12 and abs(r["c_prime"] - cp) < 1e-12
    assert abs(r["indirect"] - (c - cp)) < 1e-12
