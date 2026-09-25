"""Tests for lof.local_outlier_factor (Breunig et al. 2000)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.lof import local_outlier_factor


def _lof(X, k):
    n = len(X)
    D = [[math.dist(a, b) for b in X] for a in X]
    nb = [sorted((j for j in range(n) if j != i), key=lambda j: D[i][j])[:k] for i in range(n)]
    kd = [D[i][nb[i][-1]] for i in range(n)]
    lrd = [1.0 / (sum(max(kd[o], D[p][o]) for o in nb[p]) / k) for p in range(n)]
    return [sum(lrd[o] for o in nb[p]) / (k * lrd[p]) for p in range(n)]


def test_lof_basic():
    """LOF_k(p) = mean over N_k(p) of lrd(o) / lrd(p), with lrd the
    inverse mean reachability distance max(k-dist(o), d(p, o)),
    recomputed by brute force (continuous data: no distance ties)."""
    X = np.random.default_rng(42).normal(0, 1, (60, 3)).tolist()
    result = local_outlier_factor(X, 5)
    assert isinstance(result, dict)
    assert [float(v) for v in result["lof"]] == pytest.approx(_lof(X, 5), rel=1e-12)


def test_lof_edge():
    """A point far from a tight cluster gets the largest LOF; k outside
    1 .. n - 1 is refused."""
    X = np.random.default_rng(1).normal(0, 0.2, (30, 2)).tolist() + [[3.0, 3.0]]
    r = local_outlier_factor(X, 6)
    lof = [float(v) for v in r["lof"]]
    assert lof.index(max(lof)) == 30
    with pytest.raises(ValueError):
        local_outlier_factor(X, 31)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.lof as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
