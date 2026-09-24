"""Tests for kmnxtg.kamath_nextgpt_any2any."""

from morie.fn import _array_core as np

from morie.fn.kmnxtg import kamath_nextgpt_any2any


def test_kmnxtg_basic():
    """Test basic functionality."""
    inputs_by_modality = np.random.default_rng(42).normal(0, 1, 100)
    encoders = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    decoders = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmnxtg_edge():
    """Test edge cases."""
    inputs_by_modality = np.random.default_rng(42).normal(0, 1, 100)
    encoders = np.random.default_rng(42).normal(0, 1, 100)
    llm = np.random.default_rng(42).normal(0, 1, 100)
    decoders = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmnxtg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
