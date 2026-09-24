"""Tests for agmupv.muzero_predict_value."""

import math

from morie.fn import _array_core as np
from morie.fn.agmupv import muzero_predict_value


def test_agmupv_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    support = 21
    logits = rng.uniform(0, 1, 2 * support + 1)
    result = muzero_predict_value(logits, support)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])


def test_agmupv_edge():
    """Test edge cases."""
    support = 11
    logits = [0.0] * (2 * support + 1)
    logits[support] = 1.0
    result = muzero_predict_value(logits, support)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
