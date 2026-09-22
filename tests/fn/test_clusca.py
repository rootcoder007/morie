"""Tests for clusca.clustering_coefficient."""

from morie.fn import _array_core as np

from morie.fn.clusca import clustering_coefficient


def _manual_clustering(A, node=None):
    """Independent reference implementation using plain arithmetic."""
    n = len(A)
    local = []
    tri = 0.0
    trip = 0.0
    for v in range(n):
        nb = [u for u in range(n) if u != v and A[v][u] != 0.0]
        kv = len(nb)
        links = 0.0
        for a in range(kv):
            for b in range(a + 1, kv):
                if A[nb[a]][nb[b]] != 0.0:
                    links += 1.0
        tri += links
        trip += kv * (kv - 1.0) / 2.0
        local.append(2.0 * links / (kv * (kv - 1.0)) if kv > 1 else 0.0)
    good = [local[v] for v in range(n)
            if len([u for u in range(n) if u != v and A[v][u] != 0.0]) > 1]
    average = sum(good) / len(good) if good else float("nan")
    trans = tri / trip if trip > 0.0 else float("nan")
    if node is None:
        estimate = average
    else:
        estimate = local[int(node)]
    return estimate, local, average, trans, n


def test_clusca_basic():
    """Test basic functionality on a well-defined 5-vertex adjacency matrix."""
    n = 5
    A = [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]

    est_exp, local_exp, avg_exp, trans_exp, _ = _manual_clustering(A, node=2)
    result = clustering_coefficient(A, node=2)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "local" in result
    assert "average" in result
    assert "transitivity" in result

    assert result["estimate"] == est_exp
    assert list(result["local"]) == local_exp
    assert result["average"] == avg_exp
    assert result["transitivity"] == trans_exp


def test_clusca_edge():
    """Test edge cases: empty graph, node=None averaging, and arity-1 call."""
    n = 5
    A = [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]

    est_exp, _, avg_exp, trans_exp, n_exp = _manual_clustering(A)
    result = clustering_coefficient(A)

    assert isinstance(result, dict)
    assert result["estimate"] == avg_exp
    assert result["average"] == avg_exp
    assert result["transitivity"] == trans_exp
    assert result["n"] == n_exp
    assert len(result["local"]) == n

    empty = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    result_empty = clustering_coefficient(empty)
    assert isinstance(result_empty, dict)
    assert result_empty["average"] != result_empty["average"]
    assert result_empty["transitivity"] != result_empty["transitivity"]
    assert len(result_empty["local"]) == 3
