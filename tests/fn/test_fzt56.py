"""Tests for fzt56.fauzi_thm5_6_bdfree_ks_equiv."""

from morie.fn import _array_core as np

from morie.fn.fzt56 import fauzi_thm5_6_bdfree_ks_equiv


def test_fzt56_basic():
    """Test basic functionality."""
    ks_n = 0.12
    ks_smoothed = 0.10
    tol = 0.05
    expected_difference = abs(ks_n - ks_smoothed)
    result = fauzi_thm5_6_bdfree_ks_equiv(ks_n, ks_smoothed, tol=tol)
    assert isinstance(result, dict)
    assert "difference" in result
    assert "close" in result
    assert "tol" in result
    assert "bwok" in result
    assert "method" in result
    assert result["difference"] == expected_difference
    assert result["close"] == (expected_difference < tol)
    assert result["tol"] == tol
    assert result["bwok"] is None
    assert isinstance(result["close"], bool)


def test_fzt56_edge():
    """Test edge cases: identical statistics within tolerance and bandwidth check."""
    ks_value = 0.07
    tol = 0.05
    # Identical empirical and smoothed statistics -> difference must be 0 and close=True.
    result_identical = fauzi_thm5_6_bdfree_ks_equiv(ks_value, ks_value, tol=tol)
    assert isinstance(result_identical, dict)
    assert result_identical["difference"] == 0.0
    assert result_identical["close"] is True
    assert result_identical["bwok"] is None

    # With bandwidth h and sample size n supplied, bwok must reflect h < n^{-1/4}.
    h = 0.1
    n = 256
    expected_bwok = bool(h < n ** -0.25)
    result_bw = fauzi_thm5_6_bdfree_ks_equiv(ks_value, ks_value, tol=tol, h=h, n=n)
    assert result_bw["bwok"] == expected_bwok

    # With h and n violating the o(n^{-1/4}) condition, bwok must be False.
    h_large = 1.0
    n_large = 100
    result_bw_bad = fauzi_thm5_6_bdfree_ks_equiv(
        ks_value, ks_value, tol=tol, h=h_large, n=n_large
    )
    assert result_bw_bad["bwok"] is False
