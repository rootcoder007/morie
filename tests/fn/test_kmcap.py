"""Tests for kmcap.kamath_expert_capacity_factor."""

from morie.fn import _array_core as np

from morie.fn.kmcap import kamath_expert_capacity_factor


def test_kmcap_basic():
    """Test basic functionality."""
    tokens_per_batch = 100
    num_experts = 4
    C = 0.5
    result = kamath_expert_capacity_factor(tokens_per_batch, num_experts, C)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcap_edge():
    """Test edge cases."""
    tokens_per_batch = 100
    num_experts = 4
    C = 0.5
    result = kamath_expert_capacity_factor(tokens_per_batch, num_experts, C)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmcap as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
