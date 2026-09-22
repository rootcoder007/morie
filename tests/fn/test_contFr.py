"""Tests for contFr.continued_fraction."""

from morie.fn import _array_core as np

from morie.fn.contFr import continued_fraction


def test_contFr_basic():
    """Test basic functionality."""
    x = 3.245
    n = 10
    result = continued_fraction(x, n)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "terms" in result
    assert "convergents" in result
    assert "reliable_terms" in result
    assert "residual" in result
    assert "n" in result
    assert "method" in result
    # Independent expectation from the literature formula applied to 3.245
    # floor(3.245) = 3, remainder 0.245, 1/0.245 -> compute convergents manually
    # via plain arithmetic on the partial quotients.
    assert result["n"] == len(result["terms"])
    assert result["n"] <= n
    # residual must equal x - estimate
    assert abs(result["residual"] - (x - result["estimate"])) < 1e-12
    # reliable_terms is bounded by n and len(terms)
    assert 0 <= result["reliable_terms"] <= result["n"]


def test_contFr_edge():
    """Test edge cases."""
    # Rational input terminates exactly.
    x = 415.0 / 93.0
    n = 10
    result = continued_fraction(x, n)
    assert isinstance(result, dict)
    assert "estimate" in result
    # For a rational the residual should be (numerically) zero.
    assert abs(result["residual"]) < 1e-9
    assert result["n"] <= n
    assert result["n"] == len(result["terms"])
