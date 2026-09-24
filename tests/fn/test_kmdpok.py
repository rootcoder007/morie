"""Tests for kmdpok.kamath_dpo_loss."""

from morie.fn import _array_core as np

from morie.fn.kmdpok import kamath_dpo_loss


def test_kmdpok_basic():
    """Test basic functionality."""
    logp_w = [-1.0, -2.0]
    logp_l = [-3.0, -1.0]
    logp_ref_w = [-2.0, -2.5]
    logp_ref_l = [-3.0, -3.0]
    beta = 2.0
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmdpok_edge():
    """Test edge cases."""
    logp_w = [-1.0, -2.0]
    logp_l = [-3.0, -1.0]
    logp_ref_w = [-2.0, -2.5]
    logp_ref_l = [-3.0, -3.0]
    beta = 2.0
    result = kamath_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmdpok as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
