"""Tests for bnseff.bound_efficient."""

from morie.fn import _array_core as np

from morie.fn.bnseff import bound_efficient


def test_bnseff_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    D = (rng.uniform(0, 1, 100) < 0.5).astype(float)
    y = 2.0 * D + rng.normal(0, 1, 100)
    result = bound_efficient(y, D, X)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "se_bound" in result
    assert "outcome_term" in result
    assert "heterogeneity_term" in result
    assert "overlap_penalty" in result
    assert "ipw_relative_efficiency" in result
    assert "n_effective" in result
    assert result["bound"] > 0
    assert result["se_bound"] > 0
    # bound should equal outcome term plus heterogeneity term
    assert np.isclose(result["bound"],
                      result["outcome_term"] + result["heterogeneity_term"])
    # both components of the bound are non-negative
    assert result["outcome_term"] >= 0
    assert result["heterogeneity_term"] >= 0


def test_bnseff_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    D = (rng.uniform(0, 1, 100) < 0.5).astype(float)
    y = 2.0 * D + rng.normal(0, 1, 100)
    result = bound_efficient(y, D, X)
    assert isinstance(result, dict)
    assert "bound" in result
    assert result["bound"] >= 0
