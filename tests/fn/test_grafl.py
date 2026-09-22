"""Tests for grafl.graphlet_kernel."""

import numpy as _np

from morie.fn import _array_core as np

from morie.fn.grafl import graphlet_kernel


def _build_graph(n_nodes, edges, seed=0):
    """Build an n_nodes x n_nodes adjacency matrix with the given undirected edges.

    The graphlet_kernel expects a NetworkX-like graph or an adjacency array;
    it indexes it with two-element tuples (neighbour info) and a 1-D array of
    node indices, which only works if the input is interpreted as an explicit
    adjacency matrix. We provide a small, hand-built matrix rather than
    random data so we can assert numerical results.
    """
    A = [[0] * n_nodes for _ in range(n_nodes)]
    for (u, v) in edges:
        A[u][v] = 1
        A[v][u] = 1
    return A


def _enumerate_all(k_size):
    """Enumerate every possible size-k induced subgraph of G.

    Returns the dict {signature: count} exactly as the implementation
    would, computed independently from the test inputs.
    """
    nodes = list(range(G_N))
    from itertools import combinations

    counts = {}
    for subset in combinations(nodes, k_size):
        # Build induced subgraph adjacency.
        local = {g: i for i, g in enumerate(subset)}
        sub = [[0] * k_size for _ in range(k_size)]
        for (u, v) in G_EDGES:
            if u in local and v in local:
                a, b = local[u], local[v]
                sub[a][b] = 1
                sub[b][a] = 1
        # Canonical signature: sorted list of rows of the adjacency.
        rows = [tuple(sub[i]) for i in range(k_size)]
        # This matches _sig's behaviour for small graphs used in the test:
        sig = tuple(sorted(rows))
        counts[sig] = counts.get(sig, 0) + 1
    return counts


# Test fixture: small path graph P4 with 4 nodes and 3 edges.
G_N = 4
G_EDGES = [(0, 1), (1, 2), (2, 3)]
G1 = _build_graph(G_N, G_EDGES)
G2 = _build_graph(G_N, G_EDGES)


def test_grafl_basic():
    """Test basic functionality on a small explicit graph."""
    k = 3
    result = graphlet_kernel(G1, G2, k)

    # Implementation returns a dict-like payload with 'estimate', 'types',
    # 'f1', 'f2' keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "f1" in result and "f2" in result
    assert "types" in result

    # Both graphs are identical, so the (normalised) frequency vectors are
    # the same and the kernel value equals sum_i f1[i]^2.
    n_types = len(result["types"])
    assert n_types > 0
    expected = sum(result["f1"][i] * result["f1"][i] for i in range(n_types))

    assert abs(result["estimate"] - expected) < 1e-12

    # Self-kernel of a non-empty graph must be positive.
    assert result["estimate"] > 0.0


def test_grafl_edge():
    """Test edge cases on the same small graph."""
    k = 3
    result = graphlet_kernel(G1, G2, k)
    assert isinstance(result, dict)
    assert "estimate" in result

    # With normalize=True (default) and non-zero graphlet counts,
    # the estimate lies in (0, 1].
    est = result["estimate"]
    assert est > 0.0
    assert est <= 1.0 + 1e-12
