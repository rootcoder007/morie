"""Tests for kmvera.kamath_vera_adapter."""

from morie.fn import _array_core as np

from morie.fn.kmvera import kamath_vera_adapter


def test_kmvera_basic():
    """Test basic functionality."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A_frozen = np.random.default_rng(42).normal(0, 1, 100)
    B_frozen = np.random.default_rng(42).normal(0, 1, 100)
    lam_b = np.random.default_rng(42).normal(0, 1, 100)
    lam_d = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmvera_edge():
    """Test edge cases."""
    W0 = np.random.default_rng(42).normal(0, 1, 100)
    A_frozen = np.random.default_rng(42).normal(0, 1, 100)
    B_frozen = np.random.default_rng(42).normal(0, 1, 100)
    lam_b = np.random.default_rng(42).normal(0, 1, 100)
    lam_d = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_vera_adapter(W0, A_frozen, B_frozen, lam_b, lam_d, x)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmvera as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
