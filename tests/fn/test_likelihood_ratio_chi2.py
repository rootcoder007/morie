"""Tests for likelihood_ratio_chi2.likelihood_ratio_chi2."""

from morie.fn import _array_core as np

from morie.fn.likelihood_ratio_chi2 import likelihood_ratio_chi2


def test_ca4e18_basic():
    """Test basic functionality."""
    neg2ll_reduced = 0.5
    neg2ll_full = 0.5
    result = likelihood_ratio_chi2(neg2ll_reduced, neg2ll_full)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca4e18_edge():
    """Test edge cases."""
    neg2ll_reduced = 0.5
    neg2ll_full = 0.5
    result = likelihood_ratio_chi2(neg2ll_reduced, neg2ll_full)
    assert isinstance(result, dict)
