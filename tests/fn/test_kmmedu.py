"""Tests for kmmedu.kamath_medusa_heads."""

from morie.fn import _array_core as np

from morie.fn.kmmedu import kamath_medusa_heads


def test_kmmedu_basic():
    """Test basic functionality."""
    hidden_state = 5
    medusa_heads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "tokens" in result


def test_kmmedu_edge():
    """Test edge cases."""
    hidden_state = 5
    medusa_heads = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = kamath_medusa_heads(hidden_state, medusa_heads, k)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmmedu as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
