"""Tests for aitclu.compositional_kmeans."""

from morie.fn import _array_core as np

from morie.fn.aitclu import compositional_kmeans


def test_aitclu_basic():
    """Test basic functionality with strictly positive compositions."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    # Compositions must be strictly positive; shift so all values > 0
    X = X - X.min(axis=0) + 1e-3
    k = 5
    result = compositional_kmeans(X, k)
    assert isinstance(result, dict)
    assert "cluster" in result
    assert "centers" in result
    assert "clr_centers" in result
    assert "withinss" in result
    assert "tot_withinss" in result
    assert "iterations" in result
    assert "n" in result
    assert "D" in result
    assert "k" in result
    assert result["n"] == 100
    assert result["D"] == 5
    assert result["k"] == 5
    # labels are 1-based, so they range over {1, ..., k}
    assert set(result["cluster"]).issubset(set(range(1, k + 1)))


def test_aitclu_edge():
    """Test edge cases with two well-separated clusters."""
    rng = np.random.default_rng(123)
    half_a = rng.normal(loc=10.0, scale=0.1, size=(50, 3))
    half_b = rng.normal(loc=0.01, scale=0.1, size=(50, 3))
    # ensure strictly positive values
    half_a = half_a - half_a.min(axis=0) + 1e-3
    half_b = half_b - half_b.min(axis=0) + 1e-3
    X = np.concatenate([half_a, half_b], axis=0)
    k = 2
    result = compositional_kmeans(X, k)
    assert isinstance(result, dict)
    assert len(result["centers"]) == k
    assert len(result["clr_centers"]) == k
    assert len(result["withinss"]) == k
    # tot_withinss equals the sum of per-cluster withinss
    assert result["tot_withinss"] == sum(result["withinss"])
    # tot_withinss must be non-negative
    assert result["tot_withinss"] >= 0
