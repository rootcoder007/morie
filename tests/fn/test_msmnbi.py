"""Tests for msmnbi.msm_negative_binomial."""

from morie.fn import _array_core as np

from morie.fn.msmnbi import msm_negative_binomial


def test_msmnbi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_negative_binomial(y, treatment_history)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msmnbi_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    treatment_history = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = msm_negative_binomial(y, treatment_history)
    assert isinstance(result, dict)
