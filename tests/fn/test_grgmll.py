"""Tests for grgmll.geron_gmm_log_likelihood."""

import math

from morie.fn import _array_core as np

from morie.fn.grgmll import geron_gmm_log_likelihood


def test_grgmll_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, d, K = 40, 3, 2
    X = rng.normal(0, 1, (n, d))
    pi = [0.5, 0.5]
    means = rng.normal(0, 1, (K, d))
    covars = np.zeros((K, d, d))
    for k in range(K):
        diag_vals = rng.uniform(0.5, 2.0, d)
        for i in range(d):
            covars[k][i][i] = diag_vals[i]
    result = geron_gmm_log_likelihood(X, pi, means, covars)
    assert isinstance(result, dict)
    assert "log_likelihood" in result
    assert "per_sample" in result
    assert "mean_log_likelihood" in result
    assert "component_log_densities" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["log_likelihood"])
    assert result["n"] == n


def test_grgmll_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, d, K = 40, 3, 1
    X = rng.normal(0, 1, (n, d))
    pi = [1.0]
    means = rng.normal(0, 1, (K, d))
    covars = np.zeros((K, d, d))
    for k in range(K):
        diag_vals = rng.uniform(0.5, 2.0, d)
        for i in range(d):
            covars[k][i][i] = diag_vals[i]
    result = geron_gmm_log_likelihood(X, pi, means, covars)
    assert isinstance(result, dict)
    assert "log_likelihood" in result
    assert math.isfinite(result["log_likelihood"])
    assert result["n"] == n


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grgmll as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
