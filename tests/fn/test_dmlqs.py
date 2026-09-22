"""Tests for dmlqs.deepml_qsar."""

from morie.fn import _array_core as np

from morie.fn.dmlqs import deepml_qsar


def test_dmlqs_basic():
    """Test basic functionality of the directed-bond message passing."""
    # h0 maps directed bonds (v, w) to feature vectors.
    h0 = {
        (0, 1): [1.0, -2.0, 3.0],
        (1, 2): [0.5, 0.5, -1.0],
        (2, 0): [-1.0, 0.0, 2.0],
    }
    # adj maps vertices to their neighbours (used as message sources).
    adj = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }

    result = deepml_qsar(h0, adj, T=1, activation="relu", exclude_reverse=True)

    assert isinstance(result, dict)
    assert result["T"] == 1
    assert result["excluded_reverse"] is True
    assert "edge_states" in result
    assert isinstance(result["edge_states"], dict)

    # Independent recomputation of the formula for edge (0, 1).
    # Messages aggregate features from u != w into v; the reverse edge u=v is excluded.
    # For v=0, w=1: candidates are adj[0] = {1, 2} minus {v=0}, then minus {w=1}
    # because exclude_reverse=True, so only u=2 contributes: H[(2, 0)] = [-1, 0, 2].
    # h0[(0, 1)] = [1, -2, 3], activation=relu => max(0, x).
    d = 3
    H0 = {k: list(v) for k, v in h0.items()}
    m = [0.0] * d
    for u in sorted(set(adj.get(0, ())) - {0}):
        if u == 1:  # exclude_reverse
            continue
        if (u, 0) in H0:
            for a in range(d):
                m[a] += H0[(u, 0)][a]
    expected_01 = [max(0.0, H0[(0, 1)][a] + m[a]) for a in range(d)]
    # m = [-1, 0, 2]; H0[(0,1)] + m = [0, -2, 5] -> relu -> [0, 0, 5]
    assert expected_01 == [0.0, 0.0, 5.0]

    # Every reported edge state must satisfy the documented formula.
    for (v, w), state in result["edge_states"].items():
        mi = [0.0] * d
        for u in sorted(set(adj.get(v, ())) - {v}):
            if u == w:
                continue
            if (u, v) in H0:
                for a in range(d):
                    mi[a] += H0[(u, v)][a]
        for a in range(d):
            assert state[a] == max(0.0, H0[(v, w)][a] + mi[a])


def test_dmlqs_edge():
    """Test that W=None path matches the activation-only update formula."""
    h0 = {
        ("a", "b"): [1.0, 2.0],
        ("b", "a"): [-1.0, 0.5],
    }
    adj = {"a": ["b"], "b": ["a"]}

    result_no_W = deepml_qsar(h0, adj, T=2, W=None, activation="relu",
                              exclude_reverse=False)

    # Independent recomputation, T iterations, no W, no exclusion.
    H = {k: [float(x) for x in v] for k, v in h0.items()}
    d = len(next(iter(H.values())))
    H0 = {k: list(v) for k, v in H.items()}
    for _ in range(2):
        new = {}
        for (v, w) in H:
            m = [0.0] * d
            for u in sorted(set(adj.get(v, ())) - {v}):
                if (u, v) in H:
                    for a in range(d):
                        m[a] += H[(u, v)][a]
            new[(v, w)] = [max(0.0, H0[(v, w)][a] + m[a]) for a in range(d)]
        H = new

    assert isinstance(result_no_W, dict)
    assert result_no_W["T"] == 2
    assert result_no_W["excluded_reverse"] is False
    for k, v in H.items():
        assert result_no_W["edge_states"][k] == v
