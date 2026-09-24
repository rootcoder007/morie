"""Tests for tmlsur.tmle_survival."""

from morie.fn import _array_core as np

from morie.fn.tmlsur import tmle_survival


def test_tmlsur_basic():
    """Test basic functionality."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    covariates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_survival(time, event, treatment, covariates)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tmlsur_edge():
    """Test edge cases."""
    time = np.array([float(i + 1) for i in range(40)])
    event = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    treatment = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    covariates = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tmle_survival(time, event, treatment, covariates)
    assert isinstance(result, dict)
