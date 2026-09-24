"""Tests for noncentrality_lambda_f.noncentrality_lambda_f."""

from morie.fn import _array_core as np

from morie.fn.noncentrality_lambda_f import noncentrality_lambda_f


def test_ca8e5_basic():
    """Test basic functionality."""
    f = 0.5
    n_total = 0.5
    result = noncentrality_lambda_f(f, n_total)
    assert isinstance(result, dict)
    assert "value" in result


def test_ca8e5_edge():
    """Test edge cases."""
    f = 0.5
    n_total = 0.5
    result = noncentrality_lambda_f(f, n_total)
    assert isinstance(result, dict)
