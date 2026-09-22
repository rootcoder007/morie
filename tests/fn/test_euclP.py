"""Tests for euclP.polynomial_gcd."""

from morie.fn import _array_core as np

from morie.fn.euclP import polynomial_gcd


def test_euclP_basic():
    """Test basic functionality."""
    # p(x) = 1 + 2x + x^2 = (x+1)^2
    # q(x) = 1 + x = (x+1)
    # Expected GCD: x + 1 with coefficients [1, 1] in ascending order.
    p = [1.0, 2.0, 1.0]
    q = [1.0, 1.0]
    result = polynomial_gcd(p, q)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "gcd" in result
    assert "degree" in result
    assert "steps" in result
    assert "n" in result
    assert "method" in result
    # Independent expected values computed from the documented formula.
    expected_gcd = [1.0, 1.0]
    expected_degree = len(expected_gcd) - 1
    expected_n = len(expected_gcd)
    assert result["gcd"] == expected_gcd
    assert result["degree"] == expected_degree
    assert result["estimate"] == float(expected_degree)
    assert result["n"] == expected_n
    assert result["method"] == "Polynomial GCD via Euclid"


def test_euclP_edge():
    """Test edge cases: one polynomial is effectively zero (constant below tol)."""
    # q is a single coefficient with magnitude at or below tol => treated as 0.
    # gcd(p, 0) should equal p (made monic). p = 2 + 2x, monic form = 1 + x.
    p = [2.0, 2.0]
    q = [0.0]
    result = polynomial_gcd(p, q)
    assert isinstance(result, dict)
    # Expected monic normalization of p:
    lead = 2.0
    expected_gcd = [v / lead for v in p]
    expected_degree = len(expected_gcd) - 1
    expected_n = len(expected_gcd)
    assert result["gcd"] == expected_gcd
    assert result["degree"] == expected_degree
    assert result["estimate"] == float(expected_degree)
    assert result["n"] == expected_n
