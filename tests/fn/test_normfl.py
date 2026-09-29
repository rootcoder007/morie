"""Tests for normfl.normalizing_flow."""

import doctest as _doctest

import morie.fn.normfl as _doctest_module
from morie.fn import _array_core as np
from morie.fn.normfl import normalizing_flow


def test_normfl_basic():
    """Test basic functionality."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow(base, flow)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_normfl_edge():
    """Test edge cases."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    flow = np.random.default_rng(42).normal(0, 1, 100)
    result = normalizing_flow(base, flow)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False, optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS
    )
    assert res.attempted > 0
    assert res.failed == 0


def test_zero_iteration_flow_is_the_standardised_normal():
    """With n_iter = 0 the maps are the identity up to the tiny tanh weights,
    so the density at the mean is close to phi(0)/sd and integrates to ~1."""
    import math

    import pytest

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 4.2, 2.5]
    m = sum(x) / 8
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / 7)
    r = normalizing_flow(x, at=[m], n_iter=0)
    assert float(r["density"][0]) == pytest.approx(1 / (sd * math.sqrt(2 * math.pi)), rel=0.05)
    full = normalizing_flow(x, n_iter=0)
    assert full["integral"] == pytest.approx(1.0, abs=0.02)
