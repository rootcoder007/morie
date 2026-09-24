"""Tests for grtmp.geron_temperature_sampling."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.grtmp import geron_temperature_sampling


def test_grtmp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    logits = rng.normal(0.0, 1.0, 50)
    T = 1.0
    result = geron_temperature_sampling(logits, T)
    assert isinstance(result, dict)
    for key in ("probabilities", "entropy", "argmax", "perplexity",
                "estimate", "n", "method", "temperature"):
        assert key in result
    probs = result["probabilities"]
    assert len(probs) == 50
    assert abs(sum(probs) - 1.0) < 1e-9
    assert math.isfinite(result["entropy"])
    assert result["entropy"] >= 0.0
    assert result["n"] == 50
    assert 0 <= result["argmax"] < 50
    assert result["temperature"] == 1.0
    assert result["estimate"] == probs
    assert math.isfinite(result["perplexity"])
    assert result["perplexity"] >= 1.0


def test_grtmp_edge():
    """Test edge cases."""
    # T = 0 is rejected per docstring (formula divides by T).
    with pytest.raises(ValueError):
        geron_temperature_sampling([1.0, 2.0, 3.0], T=0.0)
    # Small valid case: entropy is monotone increasing in T.
    r_lo = geron_temperature_sampling([2.0, 1.0, 0.0], T=0.5)
    r_md = geron_temperature_sampling([2.0, 1.0, 0.0], T=1.0)
    r_hi = geron_temperature_sampling([2.0, 1.0, 0.0], T=2.0)
    assert r_lo["n"] == 3
    assert r_md["n"] == 3
    assert r_hi["n"] == 3
    assert r_lo["entropy"] < r_md["entropy"] < r_hi["entropy"]
    # Argmax is unchanged by temperature scaling.
    assert r_lo["argmax"] == r_md["argmax"] == r_hi["argmax"] == 0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grtmp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
