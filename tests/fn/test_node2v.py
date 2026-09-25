"""Tests for node2v.node2vec."""

import pytest

from morie.fn.node2v import alpha_pq, node2vec, transition_probabilities


ADJ = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1]}


def test_node2v_basic():
    """Sec. 3.2.2: from t = 0 at v = 1, returning to 0 costs 1/p, the
    common neighbour 2 (distance 1 from t) weighs 1, and 3 (distance 2)
    weighs 1/q; probabilities are those weights over their sum."""
    assert (alpha_pq(0, 2.0, 4.0), alpha_pq(1, 2.0, 4.0), alpha_pq(2, 2.0, 4.0)) == (0.5, 1.0, 0.25)
    r = transition_probabilities(ADJ, 0, 1, 2.0, 4.0)
    assert r["nodes"] == [0, 2, 3]
    assert r["probabilities"] == pytest.approx([0.5 / 1.75, 1.0 / 1.75, 0.25 / 1.75], rel=1e-15)


def test_node2v_edge():
    """Walks follow edges, have the requested length and start at every
    node; seeded runs repeat."""
    r = node2vec(ADJ, num_walks=3, length=6, p=1.0, q=0.5, seed=2)
    assert r["n_walks"] == 12
    for w in r["walks"]:
        assert len(w) == 6
        assert all(b in ADJ[a] for a, b in zip(w, w[1:]))
    assert sorted({w[0] for w in r["walks"]}) == [0, 1, 2, 3]
    assert node2vec(ADJ, num_walks=3, length=6, seed=2)["walks"] == \
        node2vec(ADJ, num_walks=3, length=6, seed=2)["walks"]


