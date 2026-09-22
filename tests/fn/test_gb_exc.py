"""Tests for gb_exc.gibbons_exceedance_stat."""

from morie.fn import _array_core as np

from morie.fn.gb_exc import gibbons_exceedance_stat


def test_gb_exc_basic():
    """Test basic functionality."""
    i, m, n = 5, 7, 10
    result = gibbons_exceedance_stat(i, m, n)
    assert isinstance(result, dict)
    for key in ("pmf", "pmf_j", "cdf_j", "mean", "var", "i", "m", "n", "method"):
        assert key in result

    # pmf must have length m+1 (j = 0..m) and sum to 1.
    import math
    pmf = result["pmf"]
    assert len(pmf) == m + 1
    assert abs(sum(pmf) - 1.0) < 1e-12

    # Independently compute pmf at a few j's from the documented formula and check.
    den = math.comb(m + n, n)
    for j in range(m + 1):
        expected = math.comb(m + n - i - j, m - j) * math.comb(i + j - 1, j) / den
        assert abs(pmf[j] - expected) < 1e-12

    # mean and var derived from the same independent formula must match.
    expected_mean = sum(k * math.comb(m + n - i - k, m - k) * math.comb(i + k - 1, k) / den
                        for k in range(m + 1))
    expected_var = sum(k * k * math.comb(m + n - i - k, m - k) * math.comb(i + k - 1, k) / den
                       for k in range(m + 1)) - expected_mean ** 2
    assert abs(result["mean"] - expected_mean) < 1e-12
    assert abs(result["var"] - expected_var) < 1e-12


def test_gb_exc_edge():
    """Test edge cases: passing a specific j populates pmf_j and cdf_j."""
    i, m, n = 3, 4, 6
    j = 2
    result = gibbons_exceedance_stat(i, m, n, j=j)
    assert isinstance(result, dict)

    import math
    den = math.comb(m + n, n)
    expected_pmf_j = math.comb(m + n - i - j, m - j) * math.comb(i + j - 1, j) / den
    expected_cdf_j = sum(
        math.comb(m + n - i - k, m - k) * math.comb(i + k - 1, k) / den
        for k in range(j + 1)
    )
    assert abs(result["pmf_j"] - expected_pmf_j) < 1e-12
    assert abs(result["cdf_j"] - expected_cdf_j) < 1e-12

    # Echoed metadata matches what we passed in.
    assert result["i"] == i
    assert result["m"] == m
    assert result["n"] == n
