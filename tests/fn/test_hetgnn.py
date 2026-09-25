"""Tests for hetgnn.heterogeneous_gnn (HAN, Wang et al. 2019, eqs. 1-9)."""

import math

import pytest

from morie.fn.hetgnn import heterogeneous_gnn


H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5], [-1.0, 0.2]]
EDGES = {0: [3, 4], 1: [3], 2: [4], 3: [0, 1], 4: [0, 2]}
TYPES = {0: "M", 1: "M", 2: "M", 3: "A", 4: "D"}
MPS = {"MAM": ["M", "A", "M"], "MDM": ["M", "D", "M"]}
W = [[0.5, -0.2], [0.1, 0.4]]
A = [0.3, -0.1, 0.2, 0.4]
WS = [[0.7, 0.1], [-0.3, 0.5]]
BS = [0.05, -0.1]
QS = [0.6, -0.4]
NB = {"MAM": {0: [0, 1], 1: [0, 1], 2: [2]}, "MDM": {0: [0, 2], 1: [1], 2: [0, 2]}}


def _proj(h):
    return [sum(W[o][j] * h[j] for j in range(2)) for o in range(2)]


def _node(i, nbrs):
    hi = _proj(H[i])
    e = []
    for j in nbrs:
        s = sum(a * z for a, z in zip(A, hi + _proj(H[j])))   # a^T [Wh_i || Wh_j]
        e.append(s if s >= 0 else 0.2 * s)                     # LeakyReLU, eq. (3)
    ex = [math.exp(v) for v in e]
    al = [v / sum(ex) for v in ex]
    z = [sum(al[t] * _proj(H[j])[a] for t, j in enumerate(nbrs)) for a in range(2)]
    return [v if v > 0 else math.exp(v) - 1 for v in z]        # ELU, eq. (4)


def test_hetgnn_basic():
    """Node-level attention over meta-path neighbours INCLUDING the node
    itself (eqs. 1-4), semantic importance averaged over the target
    (movie) nodes only (eq. 7), softmax (eq. 8), weighted sum (eq. 9);
    non-target nodes get no embedding."""
    r = heterogeneous_gnn(H, EDGES, TYPES, MPS, A, W, WS, BS, QS)
    per = {m: [_node(i, NB[m][i]) for i in range(3)] for m in MPS}
    w = {m: sum(sum(q * math.tanh(b + sum(WS[o][j] * z[j] for j in range(2)))
                    for o, (q, b) in enumerate(zip(QS, BS))) for z in per[m]) / 3 for m in MPS}
    tot = sum(math.exp(v) for v in w.values())
    beta = {m: math.exp(w[m]) / tot for m in MPS}
    assert r["semantic_weights"] == pytest.approx(beta, rel=1e-12)
    for i in range(3):
        want = [beta["MAM"] * per["MAM"][i][a] + beta["MDM"] * per["MDM"][i][a] for a in range(2)]
        assert r["embeddings"][i] == pytest.approx(want, rel=1e-12)
    assert r["embeddings"][3] == [0.0, 0.0] and r["embeddings"][4] == [0.0, 0.0]
    assert r["target_nodes"] == [0, 1, 2]


def test_hetgnn_edge():
    """Meta-paths starting at different node types raise."""
    with pytest.raises(ValueError):
        heterogeneous_gnn(H, EDGES, TYPES, {"MAM": ["M", "A", "M"], "AMA": ["A", "M", "A"]},
                          A, W, WS, BS, QS)
