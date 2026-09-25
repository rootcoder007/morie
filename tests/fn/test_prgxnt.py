"""Tests for prgxnt.perplexity."""

import math

from morie.fn import _array_core as np

from morie.fn.prgxnt import perplexity


def test_prgxnt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.001, 1, 100)
    log_probs = [math.log(p) for p in probs]
    N = 100
    result = perplexity(log_probs, N)
    assert isinstance(result, dict)
    assert "perplexity" in result
    assert math.isfinite(result["perplexity"])


def test_prgxnt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.001, 1, 10)
    log_probs = [math.log(p) for p in probs]
    N = 10
    result = perplexity(log_probs, N)
    assert isinstance(result, dict)
    assert "perplexity" in result
    assert math.isfinite(result["perplexity"])
