"""Tests for gb_ssj.gibbons_sign_sample_size_2."""

from morie.fn.gb_ssj import gibbons_sign_sample_size_2


def test_gb_ssj_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    p = 5
    result = gibbons_sign_sample_size_2(alpha, beta, p)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result


def test_gb_ssj_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    p = 5
    result = gibbons_sign_sample_size_2(alpha, beta, p)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.gb_ssj as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
