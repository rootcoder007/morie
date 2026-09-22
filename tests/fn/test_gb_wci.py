"""Tests for gb_wci.gibbons_concordance_signif."""

from morie.fn import _array_core as np

from morie.fn.gb_wci import gibbons_concordance_signif


def test_gb_wci_basic():
    """Test basic functionality."""
    w = 0.5
    k = 5
    n = 10
    result = gibbons_concordance_signif(w, k, n)
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "df" in result
    assert "s" in result
    assert result["k"] == k
    assert result["n"] == n
    assert result["df"] == n - 1

    # Q = k * (n - 1) * w  and  S = Q * k * n * (n + 1) / 12
    expected_q = k * (n - 1.0) * w
    expected_s = expected_q * k * n * (n + 1.0) / 12.0
    assert result["statistic"] == float(expected_q)
    assert result["s"] == float(expected_s)
    assert result["w"] == w


def test_gb_wci_edge():
    """Test edge cases."""
    # Perfect concordance: w = 1 yields the largest possible Q statistic
    w = 1.0
    k = 5
    n = 6
    result = gibbons_concordance_signif(w, k, n)
    assert isinstance(result, dict)
    assert result["statistic"] == float(k * (n - 1.0) * w)
    assert result["df"] == n - 1
    assert result["p_value"] > 0.0
    assert result["p_value"] <= 1.0
    # k <= 5 and n <= 6 both satisfy the "use exact table" flag
    assert result["table_n"] == 1

    # Independence: w = 0 yields Q = 0 and p_value = 1 (upper-tail test)
    w0 = 0.0
    result0 = gibbons_concordance_signif(w0, k, n)
    assert result0["statistic"] == 0.0
    assert result0["s"] == 0.0
