"""Tests for mTriple.triply_robust_mediation."""

from morie.fn import _array_core as np

from morie.fn.mTriple import triply_robust_mediation


def test_mTriple_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(44)
    rng_m = np.random.default_rng(45)
    rng_c = np.random.default_rng(46)

    n = 100
    p = 3
    Y = rng_y.normal(0, 1, n)
    X = (rng_x.uniform(0, 1, n) < 0.5).astype(float)
    M = 0.7 * X + rng_m.normal(0, 1, n)
    C = rng_c.normal(0, 1, (n, p))
    result = triply_robust_mediation(Y, X, M, C)
    assert isinstance(result, dict)
    for key in ("nie", "nde", "total", "se", "ci",
                "proportion_mediated", "decomposition_residual",
                "nuisance_agreement"):
        assert key in result


def test_mTriple_edge():
    """Test edge cases with no confounders."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(44)
    rng_m = np.random.default_rng(45)

    n = 60
    Y = rng_y.normal(0, 1, n)
    X = (rng_x.uniform(0, 1, n) < 0.5).astype(float)
    M = 0.5 * X + rng_m.normal(0, 1, n)
    result = triply_robust_mediation(Y, X, M, None)
    assert isinstance(result, dict)
    for key in ("nie", "nde", "total", "se", "ci"):
        assert key in result


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.mTriple as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
