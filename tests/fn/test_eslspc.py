"""Tests for eslspc.esl_spectral_cluster."""

from morie.fn import _array_core as np

from morie.fn.eslspc import esl_spectral_cluster


def test_eslspc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    A = rng.normal(0, 1, (n, n))
    # Make it a valid similarity matrix: symmetric, non-negative, zero diagonal.
    W = (A + A.T) / 2.0
    W = np.abs(W)
    W = W - np.diag(np.diag(W))
    k = 3
    result = esl_spectral_cluster(W, k)
    assert isinstance(result, dict)
    # Keys documented in the function's docstring / return spec.
    for key in ("estimate", "labels", "eigenvalues", "n_components",
                "embedding", "normalized", "n", "k", "method"):
        assert key in result
    assert isinstance(result["estimate"], (int, float))
    assert isinstance(result["labels"], list)
    assert len(result["labels"]) == n
    assert isinstance(result["eigenvalues"], list)
    assert len(result["eigenvalues"]) == k
    # embedding is row-major n x k, so ravel().size == n*k
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) == n * k
    assert isinstance(result["n_components"], int)
    assert 0 <= result["n_components"] <= n
    assert result["n"] == n
    assert result["k"] == k
    assert result["normalized"] is True
    # Labels are 0-based and lie in [0, k).
    for lab in result["labels"]:
        assert isinstance(lab, int)
        assert 0 <= lab < k
    assert result["method"] == "spectral clustering on the Laplacian embedding, k-means on rows"


def test_eslspc_edge():
    """Test edge cases: two disconnected triangles, k=2."""
    W = [[0, 1, 1, 0, 0, 0],
         [1, 0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1],
         [0, 0, 0, 1, 0, 1],
         [0, 0, 0, 1, 1, 0]]
    k = 2
    result = esl_spectral_cluster(W, k)
    assert isinstance(result, dict)
    for key in ("estimate", "labels", "eigenvalues", "n_components",
                "embedding", "normalized", "n", "k", "method"):
        assert key in result
    # The graph has two connected components, so two ~zero eigenvalues.
    assert result["n_components"] == 2
    assert result["n"] == 6
    assert result["k"] == k
    # Each triangle is its own cluster and the two clusters differ.
    labels = list(result["labels"])
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    # With two complete symmetric, the unnormalised Laplacian has an
    # eigenvalue exactly 0 (counted once), and the unnormalised
    # embedding has rank exactly 2, so k-means in that rank-2 space
    # separates the two triangles.  Verify the embedding shape.
    assert len(result["embedding"]) == 6 * k
