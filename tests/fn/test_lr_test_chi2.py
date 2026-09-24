"""Tests for lr_test_chi2.lr_test_chi2."""

from morie.fn import _array_core as np

from morie.fn.lr_test_chi2 import lr_test_chi2


def test_ca7e8_basic():
    """Test basic functionality."""
    ll_null = 0.5
    ll_full = 0.5
    result = lr_test_chi2(ll_null, ll_full)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca7e8_edge():
    """Test edge cases."""
    ll_null = 0.5
    ll_full = 0.5
    result = lr_test_chi2(ll_null, ll_full)
    assert isinstance(result, dict)
