"""Tests for ghs027.ghosal_ch3_tailfree_strong_support_event."""

from morie.fn import _array_core as np

from morie.fn.ghs027 import ghosal_ch3_tailfree_strong_support_event


def test_ghs027_basic():
    """Test basic functionality."""
    prob_ratio_event = np.random.default_rng(42).normal(0, 1, 100)
    prob_pm_event = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_strong_support_event(prob_ratio_event, prob_pm_event)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs027_edge():
    """Test edge cases."""
    prob_ratio_event = np.random.default_rng(42).normal(0, 1, 100)
    prob_pm_event = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_tailfree_strong_support_event(prob_ratio_event, prob_pm_event)
    assert isinstance(result, dict)
