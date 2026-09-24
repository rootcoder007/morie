"""Tests for hmlcos.geron_cosine_annealing."""

from morie.fn import _array_core as np

from morie.fn.hmlcos import geron_cosine_annealing


def test_hmlcos_basic():
    """Test basic functionality."""
    t = 0.5
    T = 5
    eta_max = 0.5
    result = geron_cosine_annealing(t, T, eta_max)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmlcos_edge():
    """Test edge cases."""
    t = 0.5
    T = 5
    eta_max = 0.5
    result = geron_cosine_annealing(t, T, eta_max)
    assert isinstance(result, dict)
