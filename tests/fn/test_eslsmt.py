"""Tests for eslsmt.esl_smoothing_spline."""

from morie.fn import _array_core as np

from morie.fn.eslsmt import esl_smoothing_spline


def test_eslsmt_basic():
    """Test basic functionality with valid inputs."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 0.0, 2.0])
    lambda_ = 0.0
    result = esl_smoothing_spline(x, y, lambda_)
    assert isinstance(result, dict)
    # Documented return keys
    assert "estimate" in result
    assert "effective_df" in result
    assert "rss" in result
    assert "lambda" in result
    assert "n" in result
    assert "method" in result
    # At lambda_=0 the spline interpolates exactly and df equals n.
    assert result["n"] == 4
    assert result["lambda"] == 0.0
    for vi, yi in zip(result["estimate"], y):
        assert abs(vi - yi) < 1e-9
    assert abs(result["effective_df"] - 4.0) < 1e-8
    assert result["rss"] < 1e-18


def test_eslsmt_edge():
    """Test edge case: heavy penalty gives the OLS line."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 0.0, 2.0])
    # Independent OLS computation using plain arithmetic on the same x, y.
    n = 4
    sx = float(sum(x))
    sy = float(sum(y))
    sxx = float(sum(xi * xi for xi in x))
    sxy = float(sum(xi * yi for xi, yi in zip(x, y)))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    expected_line = [intercept + slope * xi for xi in x]

    result = esl_smoothing_spline(x, y, 1e10)
    assert isinstance(result, dict)
    assert len(result["estimate"]) == 4
    for got, want in zip(result["estimate"], expected_line):
        assert abs(got - want) < 1e-3
    # Heavy penalty collapses the spline to a line: df -> 2.
    assert abs(result["effective_df"] - 2.0) < 1e-3
