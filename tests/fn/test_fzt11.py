"""Tests for fzt11.fauzi_thm1_1_bias_mgkde."""

from morie.fn import _array_core as np

from morie.fn.fzt11 import fauzi_thm1_1_bias_mgkde


def test_fzt11_basic():
    """Test basic functionality."""
    x = 1.0
    fp = -0.2
    fpp = 0.1
    f = 0.4
    n = 200
    bandwidth = 0.3
    result = fauzi_thm1_1_bias_mgkde(x, bandwidth, n, fp, fpp, f)
    assert isinstance(result, dict)
    assert "bias" in result
    assert "variance" in result
    assert "mse" in result
    assert "region" in result
    assert result["region"] == "interior"
    # Independent check using the documented formula.
    rh = float(np.sqrt(bandwidth))
    expected_bias = (fp + 0.5 * x * x * fpp) * rh
    assert abs(result["bias"] - expected_bias) < 1e-12
    assert abs(result["mse"] - (expected_bias * expected_bias + result["variance"])) < 1e-12


def test_fzt11_edge():
    """Test edge cases."""
    x = 1.0
    fp = -0.2
    fpp = 0.1
    f = 0.4
    n = 200
    bandwidth = 0.3
    result = fauzi_thm1_1_bias_mgkde(x, bandwidth, n, fp, fpp, f, boundary=True, c=2.0)
    assert isinstance(result, dict)
    assert "bias" in result
    assert "variance" in result
    assert "mse" in result
    assert result["region"] == "boundary"
    # Bias is the same in both regions; independent arithmetic check.
    rh = float(np.sqrt(bandwidth))
    expected_bias = (fp + 0.5 * x * x * fpp) * rh
    assert abs(result["bias"] - expected_bias) < 1e-12
