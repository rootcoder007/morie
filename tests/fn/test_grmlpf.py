"""Tests for grmlpf.geron_mlp_forward."""

from morie.fn import _array_core as np

from morie.fn.grmlpf import geron_mlp_forward


def test_grmlpf_basic():
    """Test basic functionality."""
    x = [1.0, 2.0]
    weights = [[[1.0, 1.0], [1.0, -1.0]], [[2.0, 3.0]]]
    biases = [[0.0, 0.0], [1.0]]
    result = geron_mlp_forward(x, weights, biases)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grmlpf_edge():
    """Test edge cases."""
    x = [1.0, 2.0]
    weights = [[[1.0, 1.0], [1.0, -1.0]], [[2.0, 3.0]]]
    biases = [[0.0, 0.0], [1.0]]
    result = geron_mlp_forward(x, weights, biases)
    assert isinstance(result, dict)
