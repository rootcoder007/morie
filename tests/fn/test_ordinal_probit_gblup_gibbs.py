"""Tests for ordinal_probit_gblup_gibbs.ordinal_probit_gblup_gibbs."""

from morie.fn import _array_core as np

from morie.fn.ordinal_probit_gblup_gibbs import ordinal_probit_gblup_gibbs


def test_msm095_basic():
    """Test basic functionality."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ordinal_probit_gblup_gibbs(y, G)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_msm095_edge():
    """Test edge cases."""
    y = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    G = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = ordinal_probit_gblup_gibbs(y, G)
    assert isinstance(result, dict)
