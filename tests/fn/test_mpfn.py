"""Tests for mpfn.message_passing."""

import math

from morie.fn import _array_core as np

from morie.fn.mpfn import message_passing


def test_mpfn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 5
    d = 4
    H0 = rng.normal(0, 1, (n, d))
    adj = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1, 3],
        3: [2, 4],
        4: [3],
    }
    edge_features = {(v, w): 1.0 for v in adj for w in adj[v]}

    result = message_passing(H0, adj, edge_features)

    assert isinstance(result, list)
    assert len(result) == n
    for row in result:
        assert len(row) == d
        for val in row:
            assert math.isfinite(val)


def test_mpfn_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 4
    d = 3
    H0 = rng.normal(0, 1, (n, d))
    # Isolated-node graph: every node has no neighbors.
    adj = {i: [] for i in range(n)}
    edge_features = {}

    result = message_passing(H0, adj, edge_features, T=1)

    assert isinstance(result, list)
    assert len(result) == n
    for row in result:
        assert len(row) == d
        for val in row:
            assert math.isfinite(val)
