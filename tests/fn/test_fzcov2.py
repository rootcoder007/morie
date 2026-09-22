"""Tests for fzcov2.fauzi_cov_surv_est2."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fzcov2 import fauzi_cov_surv_est2


def test_fzcov2_basic():
    """Test basic functionality."""
    n = 100
    surv = 0.6
    result = fauzi_cov_surv_est2(n, surv)
    assert isinstance(result, dict)
    assert "covariance" in result
    assert "surv" in result
    assert "cdf" in result
    assert "n" in result
    assert "method" in result
    expected_cdf = 1.0 - surv
    expected_cov = surv * expected_cdf / n
    assert result["covariance"] == expected_cov
    assert result["surv"] == surv
    assert result["cdf"] == expected_cdf
    assert result["n"] == n


def test_fzcov2_edge():
    """Test edge cases: default cdf and explicit cdf."""
    n = 50
    surv = 0.3
    # default cdf
    result = fauzi_cov_surv_est2(n, surv)
    assert isinstance(result, dict)
    expected_cdf = 1.0 - surv
    expected_cov = surv * expected_cdf / n
    assert result["covariance"] == expected_cov
    assert result["cdf"] == expected_cdf

    # explicit cdf
    cdf = 0.4
    result2 = fauzi_cov_surv_est2(n, surv, cdf=cdf)
    assert result2["covariance"] == surv * cdf / n
    assert result2["cdf"] == cdf
    assert result2["surv"] == surv
    assert result2["n"] == n
