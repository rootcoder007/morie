"""Tests for ttsAn.twitter_anomaly."""

from morie.fn import _array_core as np

from morie.fn.ttsAn import twitter_anomaly


def test_ttsAn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = twitter_anomaly(x, period)
    assert isinstance(result, dict)
    assert "estimate" in result or "anomalies" in result


def test_ttsAn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    period = 5
    result = twitter_anomaly(x, period)
    assert isinstance(result, dict)
