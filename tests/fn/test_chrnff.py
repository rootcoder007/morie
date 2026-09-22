"""Tests for chrnff.chernoff_bound."""

from morie.fn import _array_core as np

from morie.fn.chrnff import chernoff_bound


def test_chrnff_basic():
    """Test basic functionality."""
    # For X ~ N(0,1), MGF is E[exp(sX)] = exp(s^2/2).
    # Chernoff bound on P(X >= a): min_{s>0} exp(-s*a) * exp(s^2/2).
    # Analytic optimum: s = a, giving exp(-a^2/2).

    def mgf(s):
        return float(np.exp(0.5 * s * s))

    a = 1.5
    result = chernoff_bound(mgf, a)
    assert isinstance(result, dict)
    # Documented return keys.
    assert "bound" in result
    assert "s" in result
    assert "log_bound" in result
    assert "at_boundary" in result

    # The analytic minimiser is s = a.
    assert abs(result["s"] - a) < 0.1
    # Independently computed expected bound from the formula.
    expected_bound = float(np.exp(-0.5 * a * a))
    assert abs(result["bound"] - expected_bound) < 0.01
    # log_bound should match log of bound (independent computation).
    assert abs(result["log_bound"] - float(np.log(expected_bound))) < 0.01
    # The grid covers (0.01, 8.7] geometrically; a = 1.5 is interior.
    assert result["at_boundary"] is False


def test_chrnff_edge():
    """Test edge cases."""
    # Trivial mgf: E[exp(sX)] = 1 for any s (e.g. X identically 0).
    # Then bound is exp(-s*a) minimised by s -> +inf; the grid endpoint
    # should be selected and at_boundary flagged True.
    def mgf(s):
        return 1.0

    a = 2.0
    result = chernoff_bound(mgf, a)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "s" in result
    assert "log_bound" in result
    assert "at_boundary" in result
    # With mgf == 1, the bound is exp(-s*a), strictly decreasing in s,
    # so the optimum sits at the rightmost grid endpoint and is flagged.
    assert result["at_boundary"] is True
