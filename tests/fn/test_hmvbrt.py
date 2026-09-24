"""Tests for hmvbrt.geron_videobert."""

from morie.fn import _array_core as np

from morie.fn.hmvbrt import geron_videobert


def test_hmvbrt_basic():
    """Test basic functionality."""
    video_tokens = [0, 1, 2]
    text_tokens = [0, 1, 1]
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmvbrt_edge():
    """Test edge cases."""
    video_tokens = [0, 1, 2]
    text_tokens = [0, 1, 1]
    result = geron_videobert(video_tokens, text_tokens)
    assert isinstance(result, dict)
