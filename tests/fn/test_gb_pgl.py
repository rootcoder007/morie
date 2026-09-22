"""Tests for gb_pgl.gibbons_page_exact."""

from morie.fn import _array_core as np

from morie.fn.gb_pgl import gibbons_page_exact


def test_gb_pgl_basic():
    """Test basic functionality."""
    k = 5
    n = 4
    result = gibbons_page_exact(k, n)
    assert hasattr(result, "keys")
    expected_keys = {"support", "pmf", "pmf_l", "sf_l", "mean", "var", "k", "n", "method"}
    assert set(result.keys()) == expected_keys
    assert result["k"] == k
    assert result["n"] == n
    assert result["method"] == "exact null distribution of Page's L (Sec. 12.3)"


def test_gb_pgl_edge():
    """Test edge cases."""
    k = 5
    n = 4
    ell = 30
    result = gibbons_page_exact(k, n, ell=ell)
    assert hasattr(result, "keys")
    assert "support" in result
    support = result["support"]
    pmf = result["pmf"]

    # pmf must sum to 1
    assert abs(sum(pmf) - 1.0) < 1e-12

    # mean and variance must be computed from the returned support/pmf
    mean_from_pmf = sum(s * p for s, p in zip(support, pmf))
    var_from_pmf = sum(s * s * p for s, p in zip(support, pmf)) - mean_from_pmf ** 2
    assert abs(result["mean"] - mean_from_pmf) < 1e-12
    assert abs(result["var"] - var_from_pmf) < 1e-12
    assert result["var"] >= 0.0

    # With ell specified, pmf_l and sf_l should be consistent with the pmf
    assert 0.0 <= result["pmf_l"] <= 1.0
    assert 0.0 <= result["sf_l"] <= 1.0
    if support[0] <= ell <= support[-1]:
        idx = ell - support[0]
        assert abs(result["pmf_l"] - pmf[idx]) < 1e-12
        assert abs(result["sf_l"] - sum(pmf[idx:])) < 1e-12
    elif ell < support[0]:
        assert result["pmf_l"] == 0.0
        assert result["sf_l"] == 1.0
    else:
        assert result["pmf_l"] == 0.0
        assert result["sf_l"] == 0.0
