"""Tests for kmstst.kamath_stereoset_bias."""

import math
import pytest

from morie.fn import _array_core as np

from morie.fn.kmstst import kamath_stereoset_bias


def test_kmstst_basic():
    """Test basic functionality with random probabilities."""
    rng = np.random.default_rng(42)
    stereo_probs = rng.uniform(0, 1, 100)
    anti_probs = rng.uniform(0, 1, 100)
    result = kamath_stereoset_bias(stereo_probs, anti_probs)
    assert isinstance(result, dict)
    for key in ("estimate", "ss_score", "n_stereotype_preferred",
                "n_anti_preferred", "n_ties", "bias_magnitude",
                "unbiased_point", "n", "method"):
        assert key in result
    assert result["n"] == 100
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["bias_magnitude"])
    assert result["bias_magnitude"] == abs(result["estimate"] - 0.5)
    assert (result["n_stereotype_preferred"] + result["n_anti_preferred"]
            + result["n_ties"] == result["n"])


def test_kmstst_edge():
    """Empty input is rejected per the docstring."""
    with pytest.raises(ValueError):
        kamath_stereoset_bias([], [])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmstst as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
