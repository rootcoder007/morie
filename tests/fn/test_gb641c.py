"""Tests for gb641c.gibbons_median_test_ci."""

from morie.fn import _array_core as np

from morie.fn.gb641c import gibbons_median_test_ci


def test_gb641c_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0.5, 1, 100)
    c = 30
    result = gibbons_median_test_ci(x, y, c)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "estimate" in result

    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    expected_lower = float(ys[c - 1] - xs[m - c])
    expected_upper = float(ys[n - c] - xs[c - 1])
    mid_x = (xs[m // 2 - 1] + xs[m // 2]) / 2.0 if m % 2 == 0 else xs[m // 2]
    mid_y = (ys[n // 2 - 1] + ys[n // 2]) / 2.0 if n % 2 == 0 else ys[n // 2]
    expected_estimate = float(mid_y - mid_x)
    assert result["lower"] == expected_lower
    assert result["upper"] == expected_upper
    assert result["estimate"] == expected_estimate
    assert result["c"] == c
    assert result["m"] == m
    assert result["n"] == n
    assert result["method"] == "median-test confidence interval for the shift"


def test_gb641c_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = 1
    result = gibbons_median_test_ci(x, y, c)
    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "estimate" in result
    assert result["c"] == c

    xs = sorted(float(v) for v in x)
    ys = sorted(float(v) for v in y)
    m = len(xs)
    n = len(ys)
    expected_lower = float(ys[c - 1] - xs[m - c])
    expected_upper = float(ys[n - c] - xs[c - 1])
    assert result["lower"] == expected_lower
    assert result["upper"] == expected_upper
