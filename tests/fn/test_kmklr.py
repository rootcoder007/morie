"""Tests for kmklr.kamath_kl_reward_shaping."""

from morie.fn import _array_core as np

from morie.fn.kmklr import kamath_kl_reward_shaping


def test_kmklr_basic():
    """Test basic functionality."""
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    kl_divergence = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmklr_edge():
    """Test edge cases."""
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    kl_divergence = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmklr as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
