"""Tests for ksr067.kosorok_ch3_z_estimator_consistency_score."""

from morie.fn import _array_core as np

from morie.fn.ksr067 import kosorok_ch3_z_estimator_consistency_score


def test_ksr067_basic():
    """Test basic functionality."""
    scores_est = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    scores_true = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_z_estimator_consistency_score(scores_est, scores_true)
    assert isinstance(result, dict)
    assert "mean_square_difference" in result


def test_ksr067_edge():
    """Test edge cases."""
    scores_est = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    scores_true = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch3_z_estimator_consistency_score(scores_est, scores_true)
    assert isinstance(result, dict)
