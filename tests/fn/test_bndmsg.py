"""Tests for bndmsg.bound_missing_outcome."""

import math

from morie.fn import _array_core as np

from morie.fn.bndmsg import bound_missing_outcome


def test_bndmsg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = (np.random.default_rng(42).uniform(0, 1, 100) < 0.7).astype(int)
    y_min = 0.0
    y_max = 100.0
    result = bound_missing_outcome(y, R, y_min, y_max)
    # The implementation returns a RichResult that behaves like a dict.
    assert isinstance(result, dict)

    # All six documented payload keys must be present.
    for key in ("estimate", "lower", "upper", "width",
                "p_observed", "mean_observed"):
        assert key in result

    # Reconstruct the quantities from the inputs using the documented
    # formula: lower = m * p + y_min * (1 - p),
    #          upper = m * p + y_max * (1 - p),
    #          width = upper - lower = (y_max - y_min) * (1 - p).
    nobs = int(sum(int(r) for r in R))
    n = len(R)
    p = nobs / n
    m = sum(float(yy) for yy, rr in zip(y, R) if int(rr) == 1) / nobs
    expected_lower = m * p + y_min * (1.0 - p)
    expected_upper = m * p + y_max * (1.0 - p)
    expected_width = expected_upper - expected_lower
    expected_estimate = 0.5 * (expected_lower + expected_upper)

    assert math.isclose(result["lower"], expected_lower, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["upper"], expected_upper, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["width"], expected_width, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["estimate"], expected_estimate, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["p_observed"], p, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["mean_observed"], m, rel_tol=1e-12, abs_tol=1e-12)

    # lower <= estimate <= upper and width == upper - lower.
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert math.isclose(result["upper"] - result["lower"],
                        result["width"], rel_tol=1e-12, abs_tol=1e-12)


def test_bndmsg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    R = (np.random.default_rng(42).uniform(0, 1, 100) < 0.7).astype(int)
    y_min = 0.0
    y_max = 100.0
    result = bound_missing_outcome(y, R, y_min, y_max)
    assert isinstance(result, dict)
    assert "lower" in result and "upper" in result and "width" in result
    # Width must equal (y_max - y_min) * P(R = 0).
    n = len(R)
    p_obs = sum(int(r) for r in R) / n
    assert math.isclose(result["width"],
                        (y_max - y_min) * (1.0 - p_obs),
                        rel_tol=1e-12, abs_tol=1e-12)
