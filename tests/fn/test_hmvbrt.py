"""Tests for hmvbrt.geron_videobert."""

from morie.fn import _array_core as np

from morie.fn.hmvbrt import geron_videobert


def test_hmvbrt_basic():
    """Test basic functionality."""
    video_tokens = [0, 1, 2]
    text_tokens = [0, 1, 1]
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmvbrt_edge():
    """Test edge cases."""
    video_tokens = [0, 1, 2]
    text_tokens = [0, 1, 1]
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmvbrt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
