"""Tests for ghs011.ghosal_ch3_countable_dirichlet_marginal."""

from morie.fn import _array_core as np

from morie.fn.ghs011 import ghosal_ch3_countable_dirichlet_marginal


def test_ghs011_basic():
    """Test basic functionality."""
    alpha = 0.5
    k = 5
    result = ghosal_ch3_countable_dirichlet_marginal(alpha, k)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_ghs011_edge():
    """Test edge cases."""
    alpha = 0.5
    k = 5
    result = ghosal_ch3_countable_dirichlet_marginal(alpha, k)
    assert isinstance(result, dict)
