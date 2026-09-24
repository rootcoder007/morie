"""Tests for grxent.geron_softmax_cross_entropy_cost."""

import math

from morie.fn import _array_core as np

from morie.fn.grxent import geron_softmax_cross_entropy_cost


def test_grxent_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n, p, K = 40, 3, 4
    X = rng_x.normal(0, 1, (n, p))
    Y = rng_y.integers(0, K, n)
    theta = rng_t.normal(0, 1, (p, K))
    result = geron_softmax_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
    for key in ("cost", "per_instance", "probabilities",
                "accuracy", "estimate", "n", "method"):
        assert key in result
    assert math.isfinite(result["cost"])
    assert result["cost"] >= 0.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] == n
    assert len(result["per_instance"]) == n
    assert len(result["probabilities"]) == n
    for row in result["probabilities"]:
        assert len(row) == K
        s = sum(row)
        assert math.isfinite(s)
    assert math.isfinite(result["estimate"])


def test_grxent_edge():
    """Test edge cases with small valid input."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n, p, K = 10, 2, 3
    X = rng_x.normal(0, 1, (n, p))
    Y = rng_y.integers(0, K, n)
    theta = rng_t.normal(0, 1, (p, K))
    result = geron_softmax_cross_entropy_cost(X, Y, theta)
    assert isinstance(result, dict)
    assert "cost" in result
    assert math.isfinite(result["cost"])
    assert result["cost"] >= 0.0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["per_instance"]) == n
    assert len(result["probabilities"]) == n


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grxent as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
