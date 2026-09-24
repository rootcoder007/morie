"""Tests for grflam.geron_flamingo_cross_modal_attn."""

from morie.fn import _array_core as np

from morie.fn.grflam import geron_flamingo_cross_modal_attn


def test_grflam_basic():
    """Test basic functionality."""
    h = 0.3
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grflam_edge():
    """Test edge cases."""
    h = 0.3
    visual_features = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    weights = np.random.default_rng(45).exponential(1, 100)
    result = geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grflam as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
