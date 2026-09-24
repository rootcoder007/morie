"""Tests for grpre.geron_precision."""

from morie.fn import _array_core as np

from morie.fn.grpre import geron_precision


def test_grpre_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).integers(0, 2, 100)
    result = geron_precision(y_true, y_pred)
    assert isinstance(result, dict)
    assert "precision" in result
    assert "tp" in result
    assert "fp" in result
    assert result["tp"] >= 0
    assert result["fp"] >= 0


def test_grpre_edge():
    """Test edge cases."""
    yt = [1, 1, 0, 0, 1]
    yp = [1, 0, 1, 0, 1]
    result = geron_precision(yt, yp)
    assert isinstance(result, dict)
    assert "precision" in result
    assert result["tp"] == 2
    assert result["fp"] == 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grpre as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
