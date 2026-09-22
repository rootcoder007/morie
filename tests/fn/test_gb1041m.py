"""Tests for gb1041m.gibbons_kw_mult_comp."""

import math

from morie.fn import _array_core as np

from morie.fn.gb1041m import gibbons_kw_mult_comp


def test_gb1041m_basic():
    """Test basic functionality with the book's Example 10.4.1."""
    # Gibbons & Chakraborti Example 10.4.1: k = 4, n_i = 10, N = 40, alpha = 0.20
    k = 4
    n_per = 10
    ns = [n_per] * k
    N = sum(ns)
    alpha = 0.20

    # Use rank means whose pairwise differences are known.  Spacing of 5
    # between adjacent groups keeps things simple.
    rank_means = [10.0, 15.0, 20.0, 25.0]

    result = gibbons_kw_mult_comp(rank_means, ns, alpha)

    # The function returns a dict-like RichResult.
    assert isinstance(result, dict)

    # Documented keys.
    assert "bound" in result
    assert "bounds" in result
    assert "diffs" in result
    assert "significant" in result
    assert "zstar" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    # Equal-n case: bound should equal z* sqrt(k(N+1)/6).
    from scipy import stats
    expected_zstar = stats.norm.ppf(1.0 - alpha / (k * (k - 1.0)))
    expected_bound = expected_zstar * math.sqrt(k * (N + 1.0) / 6.0)

    assert result["k"] == k
    assert result["n"] == N
    assert math.isclose(result["zstar"], expected_zstar, rel_tol=1e-12)
    assert math.isclose(result["bound"], expected_bound, rel_tol=1e-12)
    assert math.isclose(result["bound"], 11.125, rel_tol=1e-3)

    # Matrices must be k x k.
    assert len(result["bounds"]) == k
    assert len(result["diffs"]) == k
    assert all(len(row) == k for row in result["bounds"])
    assert all(len(row) == k for row in result["diffs"])

    # Diagonal of diffs is zero; off-diagonal entries are the absolute
    # differences of the supplied rank means.
    for i in range(k):
        assert result["diffs"][i][i] == 0.0
        for j in range(k):
            expected_diff = abs(rank_means[i] - rank_means[j])
            assert math.isclose(result["diffs"][i][j], expected_diff,
                                rel_tol=1e-12)

    # The largest pairwise difference is 15 (between groups 0 and 3),
    # which is greater than the bound of ~11.125, so that pair must be
    # flagged significant; smaller pairs must not be.
    assert isinstance(result["significant"], list)
    flagged = [tuple(p) for p in result["significant"]]
    assert (0, 3) in flagged or (3, 0) in flagged


def test_gb1041m_edge():
    """Test edge case: unequal sample sizes => equal-n bound is nan."""
    k = 3
    ns = [5, 7, 9]
    N = sum(ns)
    rank_means = [12.0, 18.0, 22.0]
    alpha = 0.10

    result = gibbons_kw_mult_comp(rank_means, ns, alpha)

    assert isinstance(result, dict)
    assert result["k"] == k
    assert result["n"] == N
    # Unequal n_i -> equal-n bound does not apply.
    assert math.isnan(result["bound"])

    # Pairwise bounds must use the general formula
    # z* sqrt(N(N+1)/12 * (1/n_i + 1/n_j)).
    from scipy import stats
    zstar = stats.norm.ppf(1.0 - alpha / (k * (k - 1.0)))
    for i in range(k):
        for j in range(k):
            expected = zstar * math.sqrt(
                N * (N + 1.0) / 12.0 * (1.0 / ns[i] + 1.0 / ns[j])
            )
            assert math.isclose(result["bounds"][i][j], expected,
                                rel_tol=1e-12)
