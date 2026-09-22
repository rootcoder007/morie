"""Tests for eslcp.esl_mallows_cp."""

from morie.fn import _array_core as np

from morie.fn.eslcp import esl_mallows_cp


def test_eslcp_basic():
    """Test basic functionality."""
    RSS = 100.0
    d = 3
    n = 50
    sigma2 = 2.0
    result = esl_mallows_cp(RSS, d, n, sigma2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "cp_classical" in result
    expected_estimate = (RSS + 2.0 * d * sigma2) / n
    expected_cp_classical = RSS / sigma2 - n + 2.0 * d
    assert result["estimate"] == expected_estimate
    assert result["cp_classical"] == expected_cp_classical
    assert result["RSS"] == float(RSS)
    assert result["d"] == int(d)
    assert result["n"] == int(n)
    assert result["sigma2"] == float(sigma2)


def test_eslcp_edge():
    """Test edge cases."""
    RSS = 100.0
    d = 5
    n = 100
    sigma2 = 1.5
    result = esl_mallows_cp(RSS, d, n, sigma2)
    assert isinstance(result, dict)
    expected_estimate = (RSS + 2.0 * d * sigma2) / n
    expected_cp_classical = RSS / sigma2 - n + 2.0 * d
    assert result["estimate"] == expected_estimate
    assert result["cp_classical"] == expected_cp_classical
    assert result["method"].startswith("C_p")
