"""Tests for grmlc.geron_classification_mlp_output."""

from morie.fn import _array_core as np

from morie.fn.grmlc import geron_classification_mlp_output


def test_grmlc_basic():
    """Test basic functionality."""
    a_last = [1.0]
    W_out = [[2.0], [0.0], [-1.0]]
    b_out = [0.0, 0.0, 0.0]
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmlc_edge():
    """Test edge cases."""
    a_last = [1.0]
    W_out = [[2.0], [0.0], [-1.0]]
    b_out = [0.0, 0.0, 0.0]
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
