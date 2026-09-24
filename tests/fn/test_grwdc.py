"""Tests for grwdc.geron_adamw_decoupled_weight_decay."""

from morie.fn import _array_core as np

from morie.fn.grwdc import geron_adamw_decoupled_weight_decay


def test_grwdc_basic():
    """Test basic functionality."""
    theta = [1.0, 10.0]
    grad = [0.1, 0.1]
    m = [0.0, 0.0]
    s = [0.0, 0.0]
    t = 1
    eta = 0.01
    result = geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grwdc_edge():
    """Test edge cases."""
    theta = [1.0, 10.0]
    grad = [0.1, 0.1]
    m = [0.0, 0.0]
    s = [0.0, 0.0]
    t = 1
    eta = 0.01
    result = geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta)
    assert isinstance(result, dict)
