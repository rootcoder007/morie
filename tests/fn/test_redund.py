"""Tests for redund.redundancy."""

import math

from morie.fn.redund import redundancy


def test_redund_basic():
    """Test basic functionality."""
    # uniform pmf over three symbols
    p = [1/3, 1/3, 1/3]
    result = redundancy(p)
    # check that result has the expected keys
    for key in ("estimate", "entropy", "hmax", "relative", "n", "method"):
        assert key in result
    # redundancy is a ratio; it must be finite and within [0, 1]
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    # n should equal length of input alphabet
    assert result["n"] == len(p)
    # for a uniform distribution, entropy equals hmax, relative = 1, redundancy = 0
    assert math.isclose(result["entropy"], result["hmax"], rel_tol=1e-9)
    assert math.isclose(result["relative"], 1.0, rel_tol=1e-9)
    # use abs_tol when comparing to 0.0 because rel_tol scales by max(|a|, |b|)
    assert math.isclose(result["estimate"], 0.0, abs_tol=1e-9)


def test_redund_edge():
    """Test edge cases."""
    # minimum alphabet size with non-uniform probabilities
    p = [0.2, 0.8]
    result = redundancy(p, base=2.0)
    for key in ("estimate", "entropy", "hmax", "relative", "n", "method"):
        assert key in result
    assert result["n"] == 2
    # hmax = log2(2) = 1
    assert math.isclose(result["hmax"], 1.0, rel_tol=1e-9)
    # compute expected entropy and redundancy
    expected_entropy = -(0.2 * math.log2(0.2) + 0.8 * math.log2(0.8))
    expected_redundancy = 1.0 - expected_entropy
    assert math.isclose(result["entropy"], expected_entropy, rel_tol=1e-9)
    assert math.isclose(result["estimate"], expected_redundancy, rel_tol=1e-9)
    assert 0.0 <= result["estimate"] <= 1.0
