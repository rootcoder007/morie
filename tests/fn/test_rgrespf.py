"""Tests for rgrespf.rangayyan_respiration_features."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_respiration_features


def test_rgrespf_basic():
    """Test basic functionality."""
    resp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_respiration_features(resp, fs)
    assert isinstance(result, dict)
    assert "rate_breaths_per_min" in result


def test_rgrespf_edge():
    """Test edge cases."""
    resp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_respiration_features(resp, fs)
    assert isinstance(result, dict)
