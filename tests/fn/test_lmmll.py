"""Tests for lmmll.lmm_loglik."""

import math

from morie.fn import _array_core as np

from morie.fn.lmmll import lmm_loglik


def test_lmmll_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(44)
    rng_D = np.random.default_rng(45)
    rng_R = np.random.default_rng(46)
    n = 100
    p = 5
    q = 3
    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, p))
    Z = rng_Z.normal(0, 1, (n, q))
    beta = rng_X.normal(0, 1, p)
    D = rng_D.normal(0, 1, (q, q))
    D = D @ D.T + np.eye(q)
    R = rng_R.normal(0, 1, (n, n))
    R = R @ R.T + np.eye(n)
    result = lmm_loglik(y, X, Z, beta, D, R)
    assert isinstance(result, dict)
    assert 'loglik' in result
    assert math.isfinite(result['loglik'])


def test_lmmll_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(44)
    rng_D = np.random.default_rng(45)
    rng_R = np.random.default_rng(46)
    n = 40
    p = 3
    q = 2
    y = rng_y.normal(0, 1, n)
    X = rng_X.normal(0, 1, (n, p))
    Z = rng_Z.normal(0, 1, (n, q))
    beta = rng_X.normal(0, 1, p)
    D = rng_D.normal(0, 1, (q, q))
    D = D @ D.T + np.eye(q)
    R = rng_R.normal(0, 1, (n, n))
    R = R @ R.T + np.eye(n)
    result = lmm_loglik(y, X, Z, beta, D, R)
    assert isinstance(result, dict)
    assert 'loglik' in result
    assert math.isfinite(result['loglik'])


# --- appended: the module's own worked example as a gate -----------
import doctest as _doctest
import morie.fn.lmmll as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
