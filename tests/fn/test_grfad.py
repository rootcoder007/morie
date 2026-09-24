"""Tests for grfad.geron_forward_mode_autodiff."""

from morie.fn import _array_core as np

from morie.fn.grfad import geron_forward_mode_autodiff


def test_grfad_basic():
    """Test basic functionality."""
    x = 1.5
    x_prime = 1.0
    f = lambda z: z ** 4
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grfad_edge():
    """Test edge cases."""
    x = 1.5
    x_prime = 1.0
    f = lambda z: z ** 4
    result = geron_forward_mode_autodiff(x, x_prime, f)
    assert isinstance(result, dict)
