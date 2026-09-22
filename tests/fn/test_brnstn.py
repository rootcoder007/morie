"""Tests for brnstn.bernstein_inequality."""

from morie.fn import _array_core as np

from morie.fn.brnstn import bernstein_inequality


def test_brnstn_basic():
    """Test basic functionality."""
    sigma2 = 1.0
    M = 2.0
    n = 100
    t = 0.5
    result = bernstein_inequality(sigma2, M, n, t)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "bound_two_sided" in result
    assert "hoeffding" in result
    assert "ratio" in result
    assert "exponent" in result
    # Independent computation of the formula: exp(-n t^2 / (2 sigma^2 + 2 M t / 3))
    expected_ex = -n * t * t / (2.0 * sigma2 + 2.0 * M * t / 3.0)
    expected_bound = np.exp(expected_ex)
    assert result["exponent"] == expected_ex
    assert result["bound"] == expected_bound
    # bound_two_sided is min(1, 2*bound) for one-sided bound <= 0.5
    assert result["bound_two_sided"] == min(1.0, 2.0 * expected_bound)
    # Hoeffding bound: exp(-n t^2 / (2 M^2))
    expected_hoef = np.exp(-n * t * t / (2.0 * M * M))
    assert result["hoeffding"] == expected_hoef
    assert result["ratio"] == expected_bound / expected_hoef


def test_brnstn_edge():
    """Test edge cases."""
    sigma2 = 0.5
    M = 1.0
    n = 50
    t = 0.1
    result = bernstein_inequality(sigma2, M, n, t)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "exponent" in result
    expected_ex = -n * t * t / (2.0 * sigma2 + 2.0 * M * t / 3.0)
    assert result["exponent"] == expected_ex
    assert result["bound"] == np.exp(expected_ex)
