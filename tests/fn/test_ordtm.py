"""Tests for ordtm.ordinal_threshold_model."""

from morie.fn import _array_core as np

from morie.fn.ordtm import ordinal_threshold_model


def test_ordtm_basic():
    """Test basic functionality."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    thresholds = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ordinal_threshold_model(eta, thresholds)
    assert isinstance(result, dict)
    assert "probabilities" in result


def test_ordtm_edge():
    """Test edge cases."""
    eta = np.random.default_rng(42).normal(0.0, 1.0, 40)
    thresholds = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = ordinal_threshold_model(eta, thresholds)
    assert isinstance(result, dict)
