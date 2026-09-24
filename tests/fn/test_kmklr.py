"""Tests for kmklr.kamath_kl_reward_shaping."""

import math

from morie.fn import _array_core as np

from morie.fn.kmklr import kamath_kl_reward_shaping


def test_kmklr_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r_phi = rng.normal(0, 1, 100)
    kl_divergence = np.abs(rng.normal(0, 1, 100))
    beta = 0.8
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)
    assert "shaped" in result
    assert "estimate" in result
    assert len(result["shaped"]) == 100
    assert math.isfinite(result["estimate"])


def test_kmklr_edge():
    """Test edge cases: broadcasting a single KL value across a batch."""
    rng = np.random.default_rng(42)
    r_phi = rng.normal(0, 1, 10)
    kl_divergence = [0.5]
    beta = 0.1
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)
    assert "shaped" in result
    assert len(result["shaped"]) == 10
    assert math.isfinite(result["estimate"])


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
