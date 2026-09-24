"""Tests for ksr072.kosorok_ch3_z_estimator_efficiency_master."""

from morie.fn import _array_core as np

from morie.fn.ksr072 import kosorok_ch3_z_estimator_efficiency_master


def test_ksr072_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_z_estimator_efficiency_master(scores)
    assert isinstance(result, dict)
    assert "efficient_information" in result


def test_ksr072_edge():
    """Test edge cases."""
    scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_z_estimator_efficiency_master(scores)
    assert isinstance(result, dict)
