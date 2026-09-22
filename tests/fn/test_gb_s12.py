"""Tests for gb_s12.gibbons_smirnov_one_sided."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_s12 import gibbons_smirnov_one_sided


def test_gb_s12_basic():
    """Test basic functionality."""
    d = 0.2
    m = 5
    n = 7
    result = gibbons_smirnov_one_sided(d, m, n)

    # Documented return keys (RichResult behaves like a dict)
    assert isinstance(result, dict)
    for key in ("sf", "cdf", "sf_asymp", "k", "m", "n", "method"):
        assert key in result

    # Sizes are echoed back
    assert result["m"] == m
    assert result["n"] == n

    # k = sqrt(m*n/(m+n)) * d, computed independently from the inputs
    expected_k = math.sqrt(m * n / float(m + n)) * d
    assert math.isclose(result["k"], expected_k, rel_tol=1e-12, abs_tol=1e-12)

    # sf_asymp = exp(-2*k^2), computed independently
    expected_sf_asymp = math.exp(-2.0 * expected_k * expected_k)
    assert math.isclose(result["sf_asymp"], expected_sf_asymp,
                        rel_tol=1e-12, abs_tol=1e-12)

    # cdf and sf are complementary and both lie in [0, 1]
    assert math.isclose(result["cdf"], 1.0 - result["sf"],
                        rel_tol=1e-12, abs_tol=1e-12)
    assert 0.0 <= result["sf"] <= 1.0
    assert 0.0 <= result["cdf"] <= 1.0


def test_gb_s12_edge():
    """Test edge cases."""
    # d == 1 is the documented upper bound of (0, 1]
    d = 1.0
    m = 3
    n = 4
    result = gibbons_smirnov_one_sided(d, m, n)

    assert isinstance(result, dict)
    for key in ("sf", "cdf", "sf_asymp", "k", "m", "n", "method"):
        assert key in result

    expected_k = math.sqrt(m * n / float(m + n)) * d
    expected_sf_asymp = math.exp(-2.0 * expected_k * expected_k)
    assert math.isclose(result["k"], expected_k, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["sf_asymp"], expected_sf_asymp,
                        rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["cdf"], 1.0 - result["sf"],
                        rel_tol=1e-12, abs_tol=1e-12)
