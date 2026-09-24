"""Tests for hmmlpf.geron_mlp."""

from morie.fn import _array_core as np

from morie.fn.hmmlpf import geron_mlp


def test_hmmlpf_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmmlpf_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    weights = np.random.default_rng(45).exponential(1, 100)
    biases = np.random.default_rng(42).normal(0, 1, 100)
    activations = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_mlp(X, weights, biases, activations)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmlpf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
