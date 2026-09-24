"""Tests for ngcf.ngcf."""

from morie.fn import _array_core as np

from morie.fn.ngcf import ngcf


def test_ngcf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_nodes, emb_dim = 10, 3

    # Initial node embeddings: shape (n_nodes, emb_dim)
    R = rng.normal(0, 1, (n_nodes, emb_dim))

    # Adjacency as a dict mapping node -> list of neighbors
    adjacency = {}
    for i in range(n_nodes):
        neighbors = [j for j in range(n_nodes) if i != j and (j - i) % n_nodes <= 2]
        adjacency[i] = neighbors

    # NGCF propagation layers: list of (W1, W2) weight matrix pairs,
    # each of shape (emb_dim, emb_dim)
    layers = [
        (rng.normal(0, 1, (emb_dim, emb_dim)),
         rng.normal(0, 1, (emb_dim, emb_dim))),
        (rng.normal(0, 1, (emb_dim, emb_dim)),
         rng.normal(0, 1, (emb_dim, emb_dim))),
    ]

    result = ngcf(R, adjacency, layers)
    assert isinstance(result, dict)


def test_ngcf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n_nodes, emb_dim = 4, 2

    # Initial node embeddings: shape (n_nodes, emb_dim)
    R = rng.normal(0, 1, (n_nodes, emb_dim))

    # Adjacency as a dict mapping node -> list of neighbors
    adjacency = {}
    for i in range(n_nodes):
        neighbors = [j for j in range(n_nodes) if i != j]
        adjacency[i] = neighbors

    # Single NGCF propagation layer: (W1, W2) weight matrix pair
    layers = [
        (rng.normal(0, 1, (emb_dim, emb_dim)),
         rng.normal(0, 1, (emb_dim, emb_dim))),
    ]

    result = ngcf(R, adjacency, layers)
    assert isinstance(result, dict)
