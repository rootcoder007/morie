"""Tests for eslwgt.esl_weight_decay."""

from morie.fn import _array_core as np

from morie.fn.eslwgt import esl_weight_decay


def test_eslwgt_basic():
    """Test basic functionality."""
    weights = np.random.default_rng(45).exponential(1, 100)
    result = esl_weight_decay(weights)
    assert isinstance(result, dict)
    assert "penalty" in result
def test_eslwgt_edge():
    """Test edge cases."""
    weights = np.random.default_rng(45).exponential(1, 100)
    result = esl_weight_decay(weights)
    assert isinstance(result, dict)
