"""Tests for kmmae.kamath_multimodal_mae."""

from morie.fn import _array_core as np

from morie.fn.kmmae import kamath_multimodal_mae


def test_kmmae_basic():
    """Test basic functionality."""
    x_visible = np.random.default_rng(42).normal(0, 1, 100)
    x_masked_true = np.random.default_rng(42).normal(0, 1, 100)
    masks = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_multimodal_mae(x_visible, x_masked_true, masks)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmmae_edge():
    """Test edge cases."""
    x_visible = np.random.default_rng(42).normal(0, 1, 100)
    x_masked_true = np.random.default_rng(42).normal(0, 1, 100)
    masks = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_multimodal_mae(x_visible, x_masked_true, masks)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmmae as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
