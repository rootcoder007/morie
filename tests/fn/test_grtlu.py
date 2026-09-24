"""Tests for grtlu.geron_threshold_logic_unit."""

from morie.fn import _array_core as np

import pytest

from morie.fn.grtlu import geron_threshold_logic_unit


def test_grtlu_basic():
    """Test basic functionality with a 2‑D design matrix and scalar bias."""
    rng = np.random.default_rng(42)
    m, p = 40, 3
    x = rng.normal(0, 1, (m, p))      # shape (m, p)
    w = rng.normal(0, 1, p)           # shape (p,)
    b = 0.0                           # scalar bias
    result = geron_threshold_logic_unit(x, w, b)

    # The function returns a RichResult (dict‑like) with the documented payload keys
    assert isinstance(result, dict)
    for key in ("output", "margin", "estimate", "n", "method"):
        assert key in result

    out = result["output"]
    assert isinstance(out, list)
    assert len(out) == m
    assert all(v in (0, 1) for v in out)

    margin = result["margin"]
    assert isinstance(margin, list)
    assert len(margin) == m

    assert result["n"] == m
    assert isinstance(result["method"], str)


def test_grtlu_edge():
    """Edge case: passing a mismatched w length raises ValueError."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5)      # 1‑D vector of length 5
    w = rng.normal(0, 1, 3)      # wrong length → must raise
    b = 0.0
    with pytest.raises(ValueError):
        geron_threshold_logic_unit(x, w, b)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grtlu as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
