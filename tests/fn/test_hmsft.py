"""Tests for hmsft.geron_sft."""

from morie.fn import _array_core as np

from morie.fn.hmsft import geron_sft


def test_hmsft_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    instruction_data = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft(model, instruction_data, epochs, lr)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmsft_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    instruction_data = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft(model, instruction_data, epochs, lr)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmsft as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
