"""Tests for cumulative_logit.cumulative_logit."""

import math

from morie.fn import _array_core as np

from morie.fn.cumulative_logit import cumulative_logit


def test_ca5e7_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.1, 0.9, 10)
    s = sum(raw)
    probs = [x / s for x in raw]
    m = 3
    result = cumulative_logit(probs, m)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))


def test_ca5e7_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.05, 0.95, 5)
    s = sum(raw)
    probs = [x / s for x in raw]
    m = 1
    result = cumulative_logit(probs, m)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(float(result["value"]))
