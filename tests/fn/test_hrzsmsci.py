"""Tests for hrzsmsci.horowitz_sms_confidence."""

from morie.fn import _array_core as np

from morie.fn.hrzsmsci import horowitz_sms_confidence


def test_hrzsmsci_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    p = 3
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_sms_confidence(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hrzsmsci_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 20
    p = 2
    x = rng.normal(0, 1, (n, p))
    y = rng.integers(0, 2, n)
    bandwidth = 0.3
    alpha = 0.05
    result = horowitz_sms_confidence(x, y, bandwidth, alpha)
    assert isinstance(result, dict)
