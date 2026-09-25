"""Tests for gsageemd.graphsage (GraphSAGE forward pass, Algorithm 1)."""

import math

import pytest

from morie.fn.gsageemd import graphsage, sample_neighbors

# a 4-node path 0-1-2-3 with 2-D features
ADJ = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
X = [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [0.5, 0.5]]
W1 = [[0.3, -0.2, 0.5, 0.1], [0.4, 0.6, -0.3, 0.2], [-0.1, 0.2, 0.2, 0.7]]


def _layer(H, W):
    out = []
    for v in range(len(H)):
        nb = sorted(ADJ[v])
        agg = [sum(H[u][f] for u in nb) / len(nb) for f in range(len(H[0]))]
        cat = H[v] + agg
        z = [max(0.0, sum(W[o][j] * cat[j] for j in range(len(cat)))) for o in range(len(W))]
        n = math.sqrt(sum(t * t for t in z))
        out.append(z if n <= 1e-12 else [t / n for t in z])
    return out


def test_gsageemd_basic():
    """One mean-aggregator layer: h_v = normalise(relu(W [h_v || mean h_u]))."""
    r = graphsage(X, ADJ, [W1])
    want = _layer(X, W1)
    for got_row, want_row in zip(r["embeddings"], want):
        for g, w in zip(got_row, want_row):
            assert g == pytest.approx(w, abs=1e-15)
    assert r["depth"] == 1


def test_gsageemd_edge():
    """Two layers reach two hops; sampling draws only real neighbours."""
    W2 = [[0.5, -0.1, 0.3, 0.2, 0.1, -0.4], [0.2, 0.2, -0.5, 0.1, 0.3, 0.3]]
    r = graphsage(X, ADJ, [W1, W2])
    want = _layer(_layer(X, W1), W2)
    for got_row, want_row in zip(r["embeddings"], want):
        for g, w in zip(got_row, want_row):
            assert g == pytest.approx(w, abs=1e-15)
    from morie.fn import _array_core as np
    s = sample_neighbors(ADJ, 1, 5, np.random.default_rng(0))
    assert len(s) == 5 and set(s) <= {0, 2}
