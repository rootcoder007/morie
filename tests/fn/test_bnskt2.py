"""Tests for bnskt2.bound_kink_te."""

from morie.fn import _array_core as np

from morie.fn.bnskt2 import bound_kink_te


def test_bnskt2_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    x = rng_x.normal(0, 1, 100)
    kink = 0.0
    bandwidth = 1.0
    policy_slope_change = 1.0
    result = bound_kink_te(x, y, kink, bandwidth,
                           policy_slope_change=policy_slope_change)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "tau" in result
    assert "outcome_kink" in result
    assert "policy_kink" in result
    assert result["policy_kink"] == float(policy_slope_change)
    assert result["denominator_source"] == "known policy rule"
    assert result["bandwidth"] == float(bandwidth)
    assert result["order"] == 2
    assert result["kernel"] == "triangular"
    assert result["fuzzy"] is False


def test_bnskt2_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    y = rng_y.normal(0, 1, 100)
    x = rng_x.normal(0, 1, 100)
    kink = 0.0
    bandwidth = 1.0
    policy_slope_change = 2.5
    result = bound_kink_te(x, y, kink, bandwidth,
                           policy_slope_change=policy_slope_change)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["policy_kink"] == float(policy_slope_change)
    # Outcome kink must equal the ratio of right/left slope differences
    # written out independently from the documented keys.
    assert result["outcome_kink"] == (
        result["slope_right"] - result["slope_left"]
    )
    # And the estimate is that kink divided by the policy slope change.
    expected_estimate = (
        (result["slope_right"] - result["slope_left"])
        / float(policy_slope_change)
    )
    assert result["estimate"] == expected_estimate
