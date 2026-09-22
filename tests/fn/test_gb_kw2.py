"""Tests for gb_kw2.gibbons_kw_alt_form."""

from morie.fn import _array_core as np

from morie.fn.gb_kw2 import gibbons_kw_alt_form


def test_gb_kw2_basic():
    """Test basic functionality with a simple, verifiable example."""
    # Hand-crafted example with known rank sums and sample sizes.
    # Three groups of sizes n = [3, 3, 4], N = 10.
    rank_sums = [10.0, 17.0, 38.0]
    ns = [3, 3, 4]
    result = gibbons_kw_alt_form(rank_sums, ns)

    # The function returns a RichResult (mapping-like) with these keys.
    assert "statistic" in result
    assert "h_computing" in result
    assert "resid" in result
    assert "df" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    N = sum(ns)
    k = len(ns)

    # Independent computation of eq. (10.4.2).
    expected_h1 = 12.0 / (N * (N + 1.0)) * sum(
        (rank_sums[i] - ns[i] * (N + 1.0) / 2.0) ** 2 / ns[i]
        for i in range(k)
    )
    # Independent computation of eq. (10.4.7).
    expected_h2 = (
        12.0 / (N * (N + 1.0)) * sum(
            rank_sums[i] ** 2 / ns[i] for i in range(k)
        )
        - 3.0 * (N + 1.0)
    )

    assert result["statistic"] == expected_h1
    assert result["h_computing"] == expected_h2
    assert result["resid"] == expected_h1 - expected_h2
    assert result["df"] == k - 1
    assert result["k"] == k
    assert result["n"] == N


def test_gb_kw2_edge():
    """Test edge case: k = 2 (minimum number of groups)."""
    rank_sums = [6.0, 15.0]
    ns = [3, 4]
    result = gibbons_kw_alt_form(rank_sums, ns)

    assert "statistic" in result
    assert "h_computing" in result
    assert "resid" in result
    assert "df" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    N = sum(ns)
    k = len(ns)

    expected_h1 = 12.0 / (N * (N + 1.0)) * sum(
        (rank_sums[i] - ns[i] * (N + 1.0) / 2.0) ** 2 / ns[i]
        for i in range(k)
    )
    expected_h2 = (
        12.0 / (N * (N + 1.0)) * sum(
            rank_sums[i] ** 2 / ns[i] for i in range(k)
        )
        - 3.0 * (N + 1.0)
    )

    assert result["statistic"] == expected_h1
    assert result["h_computing"] == expected_h2
    assert result["resid"] == expected_h1 - expected_h2
    assert result["df"] == k - 1
    assert result["k"] == k
    assert result["n"] == N
