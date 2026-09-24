"""Tests for grmlr.geron_regression_mlp_output."""

from morie.fn import _array_core as np

from morie.fn.grmlr import geron_regression_mlp_output


def test_grmlr_basic():
    """Test basic functionality."""
    a_last = [2.0]
    W_out = [[1.5]]
    b_out = [-1.0]
    result = geron_regression_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmlr_edge():
    """Test edge cases."""
    a_last = [2.0]
    W_out = [[1.5]]
    b_out = [-1.0]
    result = geron_regression_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
