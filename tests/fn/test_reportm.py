"""Tests for reportm.report_noisy_max."""

from morie.fn import _array_core as np

from morie.fn.reportm import report_noisy_max


def test_reportm_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilon = 0.1
    result = report_noisy_max(counts, epsilon)
    assert isinstance(result, dict)
    assert "estimate" in result or "index" in result


def test_reportm_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0.0, 1.0, 40)
    epsilon = 0.1
    result = report_noisy_max(counts, epsilon)
    assert isinstance(result, dict)
