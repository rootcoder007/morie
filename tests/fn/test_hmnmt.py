"""Tests for hmnmt.geron_encoder_decoder_nmt."""

from morie.fn import _array_core as np

from morie.fn.hmnmt import geron_encoder_decoder_nmt


def test_hmnmt_basic():
    """Test basic functionality."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_nmt(src, tgt, model)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmnmt_edge():
    """Test edge cases."""
    src = np.random.default_rng(42).normal(0, 1, 100)
    tgt = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_encoder_decoder_nmt(src, tgt, model)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmnmt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
