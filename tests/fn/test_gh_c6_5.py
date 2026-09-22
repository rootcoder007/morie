"""Tests for gh_c6_5.ghosal_schwartz_thm."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_5 import ghosal_schwartz_thm


def test_gh_c6_5_basic():
    """Test basic functionality."""
    prior_mass = 0.5
    kl_radius = 0.1
    test_rate = 0.5
    n = 100
    result = ghosal_schwartz_thm(prior_mass, kl_radius, test_rate, n)
    assert isinstance(result, dict)
    assert "holds" in result
    assert "margin" in result
    assert "bound" in result
    assert "prior_mass" in result
    assert "n" in result
    assert result["holds"] == 1.0
    margin = test_rate - kl_radius
    expected_bound = np.exp(-margin * n) / prior_mass
    assert abs(result["margin"] - margin) < 1e-12
    assert abs(result["bound"] - expected_bound) < 1e-12


def test_gh_c6_5_edge():
    """Test edge cases."""
    prior_mass = 1.0
    kl_radius = 0.0
    test_rate = 0.0
    n = 1
    result = ghosal_schwartz_thm(prior_mass, kl_radius, test_rate, n)
    assert isinstance(result, dict)
    assert "holds" in result
    assert "margin" in result
    assert "bound" in result
    assert result["holds"] == 0.0
    assert result["margin"] == 0.0
    assert result["bound"] == 1.0
