"""Tests for fzt57.fauzi_thm5_7_bdfree_cvm_equiv."""

from morie.fn import _array_core as np

from morie.fn.fzt57 import fauzi_thm5_7_bdfree_cvm_equiv


def test_fzt57_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    empirical = 0.12
    smoothed = 0.14
    tol = 0.05
    result = fauzi_thm5_7_bdfree_cvm_equiv(empirical, smoothed, tol=tol)
    assert isinstance(result, dict)
    expected_diff = abs(float(empirical) - float(smoothed))
    assert result["difference"] == expected_diff
    assert result["difference"] < tol
    assert result["close"] is True
    assert result["tol"] == float(tol)
    assert result["bwok"] is None
    assert result["method"] == "boundary-free vs empirical CvM equivalence (Theorem 5.7)"


def test_fzt57_edge():
    """Test edge cases: difference exceeds tolerance and bandwidth condition."""
    # Difference exceeds tolerance
    empirical = 0.5
    smoothed = 0.1
    tol = 0.05
    result = fauzi_thm5_7_bdfree_cvm_equiv(empirical, smoothed, tol=tol)
    assert isinstance(result, dict)
    expected_diff = abs(float(empirical) - float(smoothed))
    assert result["difference"] == expected_diff
    assert result["difference"] >= tol
    assert result["close"] is False
    assert result["tol"] == float(tol)
    assert result["bwok"] is None

    # Bandwidth condition satisfied: h < n^{-1/4}
    n = 10000
    h_ok = float(n) ** -0.25 / 2.0  # half the threshold, so condition holds
    result_ok = fauzi_thm5_7_bdfree_cvm_equiv(0.01, 0.01, tol=0.05, h=h_ok, n=n)
    expected_bwok = bool(float(h_ok) < float(int(n)) ** -0.25)
    assert result_ok["bwok"] == expected_bwok
    assert result_ok["bwok"] is True

    # Bandwidth condition violated: h >= n^{-1/4}
    h_bad = float(n) ** -0.25 * 2.0  # double the threshold
    result_bad = fauzi_thm5_7_bdfree_cvm_equiv(0.01, 0.01, tol=0.05, h=h_bad, n=n)
    assert result_bad["bwok"] is False
