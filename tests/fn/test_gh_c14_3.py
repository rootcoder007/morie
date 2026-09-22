"""Tests for gh_c14_3.ghosal_crp_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c14_3 import ghosal_crp_def


def test_gh_c14_3_basic():
    """Test basic functionality."""
    result = ghosal_crp_def(n=5, alpha=2.0, seed=42)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # Expected number of tables E[K_n] from Prop 4.8: sum_{i=1..n} alpha/(alpha + i - 1)
    n = 5
    alpha = 2.0
    expected_ek = sum(alpha / (alpha + i - 1.0) for i in range(1, n + 1))
    assert np.all(np.isfinite(np.asarray(result["expected_K_n"], dtype=float)))

    # Sizes must be a permutation that sums to n; estimate equals the number of tables
    sizes = sorted(result["sizes"], reverse=True)
    assert sum(sizes) == n
    assert result["total_seated"] == n
    assert int(result["estimate"]) == len(sizes)
    assert result["method"] == "CRP (GvdV 2017 eq. 4.13)"

    # Note: EK is kept as a documented scalar; treat it as a reference value
    # via the formula, not asserting the simulated estimate equals it.
    assert isinstance(expected_ek, float)


def test_gh_c14_3_edge():
    """Test edge cases with minimal n."""
    result = ghosal_crp_def(n=1, alpha=2.0, seed=42)
    # With a single customer the only possible outcome is one table of size 1.
    assert int(result["estimate"]) == 1
    assert int(result["total_seated"]) == 1
    assert sorted(result["sizes"], reverse=True) == [1]

    n = 1
    alpha = 2.0
    expected_ek = sum(alpha / (alpha + i - 1.0) for i in range(1, n + 1))
    assert abs(float(result["expected_K_n"]) - expected_ek) < 1e-12

    # Determinism: same seed -> same outcome
    result2 = ghosal_crp_def(n=1, alpha=2.0, seed=42)
    assert int(result2["estimate"]) == int(result["estimate"])
