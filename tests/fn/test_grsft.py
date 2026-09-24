"""Tests for grsft.geron_sft_objective."""

from morie.fn import _array_core as np

from morie.fn.grsft import geron_sft_objective


def test_grsft_basic():
    """Test basic functionality."""
    logits = [[10.0, 0.0], [0.0, 0.0], [1.0, 3.0]]
    response_mask = [False, True, True]
    targets = [0, 1, 1]
    result = geron_sft_objective(logits, response_mask, targets)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grsft_edge():
    """Test edge cases."""
    logits = [[10.0, 0.0], [0.0, 0.0], [1.0, 3.0]]
    response_mask = [False, True, True]
    targets = [0, 1, 1]
    result = geron_sft_objective(logits, response_mask, targets)
    assert isinstance(result, dict)
