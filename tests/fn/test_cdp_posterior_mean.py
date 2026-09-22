"""Tests for cdp_posterior_mean.cdp_posterior_mean."""

from morie.fn import _array_core as np

from morie.fn.cdp_posterior_mean import cdp_posterior_mean


def test_ghs014_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    alpha_total = 10.0
    # alpha_j is the Dirichlet prior parameter vector (length k)
    alpha_j = rng.uniform(0.1, 5.0, 5)
    # counts is the observed count vector N_j (length k)
    counts = rng.integers(0, 20, 5)
    # j is the integer index of the category whose posterior mean we want
    j = 2
    result = cdp_posterior_mean(alpha_j, counts, j, alpha_total)

    # Result must be a dict-like with the documented keys
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result

    # Independent recomputation of the literature formula:
    # E(p_j | X) = (alpha_j + N_j) / (sum(alpha) + n)  (eq. 3.7)
    n_total = int(np.sum(counts))
    expected = (float(alpha_j[j]) + float(counts[j])) / (alpha_total + n_total)
    assert result["value"] == result["estimate"]
    assert abs(result["value"] - expected) < 1e-12


def test_ghs014_edge():
    """Test edge cases: zero observations and a large index."""
    rng = np.random.default_rng(42)
    alpha_total = 1.0
    alpha_j = rng.uniform(0.5, 2.0, 4)
    counts = np.zeros(4, dtype=int)
    j = 3  # boundary index (last category), with zero counts
    result = cdp_posterior_mean(alpha_j, counts, j, alpha_total)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "value" in result
    assert result["value"] == result["estimate"]

    expected = float(alpha_j[j]) / (alpha_total + 0)
    assert abs(result["value"] - expected) < 1e-12
