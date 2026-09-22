"""Tests for gh_c5_3.ghosal_cgibbs."""

from morie.fn import _array_core as np

from morie.fn.gh_c5_3 import ghosal_cgibbs


def test_gh_c5_3_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_cgibbs(x)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float).item())
    assert np.isfinite(est)
    # documented return keys per docstring / RichResult
    assert "n_clusters" in result
    assert "labels" in result
    # n_clusters equals the number of distinct labels
    assert int(result["n_clusters"]) == len(set(result["labels"]))
    # estimate is the cluster count, numerically equal to n_clusters
    assert est == float(result["n_clusters"])


def test_gh_c5_3_edge():
    """Test edge cases: single observation yields exactly one cluster."""
    data = np.array([42.0])
    n = int(np.asarray(data).size)
    # With a single datum and Gibbs starting from singletons, there is
    # exactly one observation so the algorithm must report one cluster,
    # i.e. n_clusters == n == 1.  We check both the documented keys.
    result = ghosal_cgibbs(data)
    assert "n_clusters" in result
    assert "estimate" in result
    assert int(result["n_clusters"]) == n
    assert float(result["estimate"]) == float(n)
