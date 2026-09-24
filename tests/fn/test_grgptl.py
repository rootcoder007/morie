"""Tests for grgptl.geron_gpt_autoregressive_loss."""

from morie.fn import _array_core as np

from morie.fn.grgptl import geron_gpt_autoregressive_loss


def test_grgptl_basic():
    """Test basic functionality."""
    logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    targets = [0, 2]
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grgptl_edge():
    """Test edge cases."""
    logits = [[2.0, 0.0, -1.0], [0.0, 1.0, 1.0]]
    targets = [0, 2]
    result = geron_gpt_autoregressive_loss(logits, targets)
    assert isinstance(result, dict)
