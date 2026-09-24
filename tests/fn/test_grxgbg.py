"""Tests for grxgbg.geron_xgboost_gain."""

from morie.fn import _array_core as np

from morie.fn.grxgbg import geron_xgboost_gain


def test_grxgbg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    GL = float(rng.normal(0, 1))
    HL = float(rng.normal(0, 1)) + 1.0
    GR = float(rng.normal(0, 1))
    HR = float(rng.normal(0, 1)) + 1.0
    lam = 0.1
    gamma = 1.0
    result = geron_xgboost_gain(GL, HL, GR, HR, lam, gamma)
    assert isinstance(result, dict)
    for key in ("gain", "left_score", "right_score", "parent_score",
                "left_weight", "right_weight", "should_split",
                "estimate", "n", "method"):
        assert key in result
    import math
    assert math.isfinite(result["gain"])


def test_grxgbg_edge():
    """Test edge cases: large gamma prunes the split."""
    # Worked-example style edge: strong gamma overrides any positive gain.
    result = geron_xgboost_gain(-2.0, 2.0, 2.0, 2.0, lam=1.0, gamma=2.0)
    assert isinstance(result, dict)
    import math
    assert math.isfinite(result["gain"])
    assert result["should_split"] is False
    assert result["gain"] < 0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grxgbg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
