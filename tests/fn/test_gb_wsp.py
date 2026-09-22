"""Tests for gb_wsp.gibbons_concordance_preference."""

from morie.fn import _array_core as np

from morie.fn.gb_wsp import gibbons_concordance_preference


def test_gb_wsp_basic():
    """Test basic functionality."""
    rankings = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 1.0, 3.0, 4.0, 5.0],
    ])
    result = gibbons_concordance_preference(rankings)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "preference_rank" in result
    assert "order" in result
    assert "rank_sums" in result
    assert "W" in result
    assert "rho_bar" in result
    assert "rho_by_observer" in result
    assert "expected_W_under_independence" in result
    assert "k" in result
    assert "n" in result

    k, n = 3, 5
    sums = np.array([1.0 + 1.0 + 2.0,
                     2.0 + 2.0 + 1.0,
                     3.0 + 3.0 + 3.0,
                     4.0 + 4.0 + 4.0,
                     5.0 + 5.0 + 5.0])
    expected_sums = sums
    assert np.allclose(np.asarray(result["rank_sums"], dtype=float), expected_sums)

    mu = k * (n + 1) / 2.0
    S = float(np.sum((expected_sums - mu) ** 2))
    denom = k ** 2 * (n ** 3 - n) / 12.0
    expected_W = S / denom
    assert abs(float(result["W"]) - expected_W) < 1e-12
    assert abs(float(result["expected_W_under_independence"]) - 1.0 / k) < 1e-12

    assert int(result["k"]) == k
    assert int(result["n"]) == n

    expected_pref = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.allclose(np.asarray(result["preference_rank"], dtype=float), expected_pref)


def test_gb_wsp_edge():
    """Test edge cases."""
    rankings = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
    result = gibbons_concordance_preference(rankings)
    assert int(result["n"]) == 2
    assert int(result["k"]) == 3
    assert "W" in result
    assert np.isfinite(float(result["W"]))
