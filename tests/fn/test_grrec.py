"""Tests for grrec.geron_recall."""

from morie.fn import _array_core as np

from morie.fn.grrec import geron_recall


def test_grrec_basic():
    """Test basic functionality."""
    rng_t = np.random.default_rng(43)
    rng_p = np.random.default_rng(44)
    y_true = rng_t.integers(0, 2, 100)
    y_pred = rng_p.integers(0, 2, 100)
    result = geron_recall(y_true, y_pred)
    assert isinstance(result, dict)
    assert "recall" in result
    assert "tp" in result
    assert "fn" in result
    assert math.isfinite(result["recall"])
    assert 0.0 <= result["recall"] <= 1.0


def test_grrec_edge():
    """Test edge cases."""
    rng_t = np.random.default_rng(43)
    rng_p = np.random.default_rng(44)
    y_true = rng_t.integers(0, 2, 100)
    y_pred = rng_p.integers(0, 2, 100)
    result = geron_recall(y_true, y_pred)
    assert isinstance(result, dict)
    assert "per_class" in result
    assert "method" in result
    assert len(result["per_class"]) == max(int(y_true.max()), int(y_pred.max())) + 1


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import math

import morie.fn.grrec as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
