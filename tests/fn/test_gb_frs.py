"""Tests for gb_frs.gibbons_friedman_chi2_approp."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_frs import gibbons_friedman_chi2_approp


def test_gb_frs_basic():
    """Test basic functionality against the documented formula."""
    q = 7.5
    k = 5
    n = 8

    result = gibbons_friedman_chi2_approp(q, k, n)

    # Expected moments per the docstring (Gibbons & Chakraborti, Sec. 12.2, p. 442)
    expected_df = n - 1
    expected_mean = float(n - 1)
    expected_var_exact = 2.0 * (n - 1) * (k - 1) / k
    expected_var_chi2 = 2.0 * (n - 1)
    expected_ratio = expected_var_exact / expected_var_chi2

    assert isinstance(result, dict)
    assert result["statistic"] == q
    assert result["df"] == expected_df
    assert result["mean"] == expected_mean
    assert math.isclose(result["var_exact"], expected_var_exact, rel_tol=1e-12)
    assert math.isclose(result["var_chi2"], expected_var_chi2, rel_tol=1e-12)
    assert math.isclose(result["ratio"], expected_ratio, rel_tol=1e-12)
    assert result["k"] == k
    assert result["n"] == n
    assert 0.0 < result["p_value"] < 1.0


def test_gb_frs_edge():
    """Test edge cases: minimum allowed k and n."""
    q = 1.0
    k = 2
    n = 2

    result = gibbons_friedman_chi2_approp(q, k, n)

    assert isinstance(result, dict)
    assert result["df"] == 1
    assert math.isclose(result["var_exact"], 1.0, rel_tol=1e-12)
    assert math.isclose(result["var_chi2"], 2.0, rel_tol=1e-12)
    assert math.isclose(result["ratio"], 0.5, rel_tol=1e-12)
    assert 0.0 < result["p_value"] < 1.0
