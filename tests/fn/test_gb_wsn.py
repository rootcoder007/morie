"""Tests for gb_wsn.gibbons_wsrt_normal_approx."""

from morie.fn import _array_core as np

from morie.fn.gb_wsn import gibbons_wsrt_normal_approx


def test_gb_wsn_basic():
    """Test basic functionality."""
    n = 100
    # Compute a valid T_plus: sum of 1..n = n(n+1)/2
    T_plus = float(n * (n + 1) // 2)
    result = gibbons_wsrt_normal_approx(T_plus, n)
    assert isinstance(result, dict)
    # z should equal (T_plus - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24)
    expected_mean = n * (n + 1.0) / 4.0
    expected_var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    expected_z = (T_plus - expected_mean) / np.sqrt(expected_var)
    assert np.isclose(result["z"], expected_z)
    assert np.isclose(result["mean"], expected_mean)
    assert np.isclose(result["var"], expected_var)
    assert np.isclose(result["statistic"], T_plus)
    assert result["n"] == n
    assert result["alternative"] == "two-sided"
    assert "z" in result
    assert "p_value" in result


def test_gb_wsn_edge():
    """Test edge cases."""
    # Use T_plus equal to the expected mean so z == 0 and p_value == 1.0
    n = 50
    T_plus = n * (n + 1.0) / 4.0
    result = gibbons_wsrt_normal_approx(T_plus, n)
    assert isinstance(result, dict)
    assert np.isclose(result["z"], 0.0)
    assert np.isclose(result["p_value"], 1.0)
    assert result["n"] == n
