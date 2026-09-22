"""Tests for bndcvr.bound_coverage_check."""

from morie.fn import _array_core as np

from morie.fn.bndcvr import bound_coverage_check


def test_bndcvr_basic():
    """Test basic functionality."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = lower + np.abs(np.random.default_rng(43).normal(2, 0.5, 100))
    theta_true = 0.0
    alpha = 0.05
    result = bound_coverage_check(lower, upper, theta_true, alpha)

    assert isinstance(result, dict)

    # Documented return keys from the function's docstring.
    assert "coverage" in result
    assert "nominal" in result
    assert "n_covered" in result
    assert "R" in result
    assert "p_value" in result
    assert "reject" in result
    assert "mean_width" in result

    R = len(lower)
    n_covered_expected = int(np.sum((lower <= theta_true) & (theta_true <= upper)))
    coverage_expected = n_covered_expected / float(R)
    p_expected = 1.0 - alpha
    # Binomial PMF accumulated from k=0 up to n_covered_expected.
    term = (1.0 - p_expected) ** R
    tail_expected = term
    for j in range(1, n_covered_expected + 1):
        term = term * p_expected * (R - j + 1) / ((1.0 - p_expected) * j)
        tail_expected += term
    if tail_expected > 1.0:
        tail_expected = 1.0
    mean_width_expected = float(np.sum(upper - lower) / R)

    assert result["n_covered"] == n_covered_expected
    assert result["R"] == R
    assert result["coverage"] == coverage_expected
    assert result["nominal"] == p_expected
    assert result["p_value"] == tail_expected
    assert result["mean_width"] == mean_width_expected
    assert result["reject"] == (1.0 if tail_expected < alpha else 0.0)


def test_bndcvr_edge():
    """Test edge cases."""
    lower = np.random.default_rng(42).normal(0, 1, 100)
    upper = lower + np.abs(np.random.default_rng(43).normal(2, 0.5, 100))
    theta_true = 0.0
    alpha = 0.05
    result = bound_coverage_check(lower, upper, theta_true, alpha)
    assert isinstance(result, dict)
    assert "p_value" in result
    assert "coverage" in result
