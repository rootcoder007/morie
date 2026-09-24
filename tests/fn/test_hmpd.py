"""Tests for hmpd.geron_padding."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hmpd import geron_padding


def test_hmpd_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (5, 5))
    result = geron_padding(x, 1, 1)
    assert isinstance(result, dict)
    assert "padded" in result
    assert "pad_h" in result
    assert "pad_w" in result
    assert "output_shape" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["padded"].shape == (7, 7)
    assert result["pad_h"] == (1, 1)
    assert result["pad_w"] == (1, 1)
    assert result["output_shape"] == (7, 7)


def test_hmpd_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (4, 4))
    result = geron_padding(x, kernel_size=2)
    assert isinstance(result, dict)
    assert result["pad_h"] == (0, 1)
    assert result["pad_w"] == (0, 1)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmpd as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
