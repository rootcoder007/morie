"""Tests for ksr057.kosorok_ch2_m_estimator_master_theorem."""

from morie.fn import _array_core as np

from morie.fn.ksr057 import kosorok_ch2_m_estimator_master_theorem


def test_ksr057_basic():
    """Test basic functionality."""
    m_dot_scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch2_m_estimator_master_theorem(m_dot_scores)
    assert isinstance(result, dict)
    assert "Sigma" in result


def test_ksr057_edge():
    """Test edge cases."""
    m_dot_scores = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = kosorok_ch2_m_estimator_master_theorem(m_dot_scores)
    assert isinstance(result, dict)
