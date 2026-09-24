"""Tests for km109.kamath_ch6_perplexity_leakage."""

from morie.fn import _array_core as np

import math

import pytest

from morie.fn.km109 import kamath_ch6_perplexity_leakage


def test_km109_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    seqs = ["w" + str(i) for i in range(10)]
    PP_public = {w: float(rng.uniform(1.0, 20.0)) for w in seqs}
    PP_lm = {w: float(rng.uniform(1.0, 20.0)) for w in seqs}
    result = kamath_ch6_perplexity_leakage(seqs, PP_public, PP_lm)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "argmax" in result
    assert "per_sequence" in result
    assert "n_leaking" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["argmax"] in seqs
    assert result["n"] == len(seqs)
    assert len(result["per_sequence"]) == len(seqs)
    assert isinstance(result["n_leaking"], int)
    assert 0 <= result["n_leaking"] <= len(seqs)


def test_km109_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        kamath_ch6_perplexity_leakage([], {}, {})
