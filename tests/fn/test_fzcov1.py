"""Tests for fzcov1.fauzi_cov_surv_est1."""

from morie.fn import _array_core as np

from morie.fn.fzcov1 import fauzi_cov_surv_est1


def test_fzcov1_basic():
    """Test basic functionality.

    Eq. (4.16): Cov[S_{X,1}(t), tilde S_X(t)] = (1/n) * S_X(t) * F_X(t).
    """
    n = 50
    surv = 0.7
    cdf = 0.3
    result = fauzi_cov_surv_est1(n, surv, cdf=cdf)

    assert isinstance(result, dict)
    assert "covariance" in result
    assert "surv" in result
    assert "cdf" in result
    assert "n" in result
    assert "method" in result

    # Independent computation per Eq. (4.16)
    expected_cov = surv * cdf / n
    assert result["covariance"] == expected_cov
    assert result["surv"] == surv
    assert result["cdf"] == cdf
    assert result["n"] == n


def test_fzcov1_default_cdf():
    """When cdf is omitted, cdf defaults to 1 - surv."""
    n = 100
    surv = 0.4
    result = fauzi_cov_surv_est1(n, surv)

    assert isinstance(result, dict)
    assert result["surv"] == surv
    assert result["cdf"] == 1.0 - surv
    assert result["n"] == n

    expected_cov = surv * (1.0 - surv) / n
    assert result["covariance"] == expected_cov


def test_fzcov1_edge():
    """Test edge cases (small n, edge surv values)."""
    n = 1
    surv = 1.0
    cdf = 0.0
    result = fauzi_cov_surv_est1(n, surv, cdf=cdf)

    assert isinstance(result, dict)
    assert result["covariance"] == surv * cdf / n
    assert result["surv"] == surv
    assert result["cdf"] == cdf
    assert result["n"] == n

    # surv = 0 case (cdf defaults to 1)
    n2 = 5
    surv2 = 0.0
    result2 = fauzi_cov_surv_est1(n2, surv2)
    assert result2["cdf"] == 1.0
    assert result2["covariance"] == 0.0 * 1.0 / n2
