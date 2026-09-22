"""Tests for gh_c4_16.ghosal_dp_tails."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_16 import ghosal_dp_tails


def test_gh_c4_16_basic():
    """Test basic functionality."""
    m = 0.1
    result = ghosal_dp_tails(m)
    assert "estimate" in result
    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))
    # Independent computation of the upper bound from the documented formula
    # exp(-1/(MG(x) |log MG(x)|^r)) with r=2, MG(x)=m
    import math
    r = 2.0
    ll = abs(math.log(m))
    expected_hi = math.exp(-1.0 / (m * ll ** r))
    assert np.all(np.isclose(est, expected_hi))


def test_gh_c4_16_edge():
    """Test edge cases."""
    m = 0.5
    result = ghosal_dp_tails(np.array([m]))
    assert "estimate" in result
    assert result["method"] == "DP tail bounds (GvdV 2017 eq. 4.24)"
    # Thinner-than-base assertion from the documented behaviour
    assert result["thinner_than_base"] is True or result["thinner_than_base"] == True
