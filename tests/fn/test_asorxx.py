"""Tests for asorxx.assortativity."""

from morie.fn import _array_core as np

from morie.fn.asorxx import assortativity


def test_asorxx_basic():
    """Test basic functionality."""
    # 5 vertices; build an undirected adjacency matrix that is NOT the identity
    # so that there is at least one edge and self-loops are ignored.
    n = 5
    G = np.zeros((n, n))
    # edges: (0,1), (0,2), (3,4)
    edges = [(0, 1), (0, 2), (3, 4)]
    for i, j in edges:
        G[i, j] = 1.0
        G[j, i] = 1.0
    # categorical attribute, one entry per vertex
    attribute = np.array(["a", "a", "a", "b", "b"])
    result = assortativity(G, attribute)
    # The function returns a RichResult-like object exposing a `.payload` dict
    payload = result.payload
    assert isinstance(payload, dict)
    assert "estimate" in payload

    # Independent recomputation of Newman's enumerative assortativity.
    types = sorted({str(v) for v in attribute}, key=lambda v: str(v))
    T = len(types)
    pos = {str(t): i for i, t in enumerate(types)}

    n_edges = len(edges)
    e = [[0.0] * T for _ in range(T)]
    for i, j in edges:
        w = G[i, j]
        ti = pos[str(attribute[i])]
        tj = pos[str(attribute[j])]
        e[ti][tj] += 0.5 * w
        e[tj][ti] += 0.5 * w
    # normalise so rows sum to 1
    for i in range(T):
        for j in range(T):
            e[i][j] = e[i][j] / n_edges

    a = [sum(e[i][j] for j in range(T)) for i in range(T)]
    b = [sum(e[j][i] for j in range(T)) for i in range(T)]
    tr = sum(e[i][i] for i in range(T))
    ab = sum(a[i] * b[i] for i in range(T))
    den = 1.0 - ab
    expected_r = (tr - ab) / den
    expected_rmin = -ab / den

    assert payload["r"] == expected_r
    assert payload["r_min"] == expected_rmin
    assert payload["trace_e"] == tr
    assert payload["sum_ab"] == ab
    assert payload["n_types"] == T
    assert payload["n"] == n
    # Sanity: all three 'a'-vertices connect only to 'a', 'b'-vertices only to 'b'
    # so r should be 1.0 (perfect assortativity by type)
    assert expected_r == 1.0


def test_asorxx_edge():
    """Test edge cases."""
    n = 5
    G = np.zeros((n, n))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    for i, j in edges:
        G[i, j] = 1.0
        G[j, i] = 1.0
    # mix of types so that ab != 1.0 and r is well defined
    attribute = np.array(["a", "a", "b", "b", "a"])
    result = assortativity(G, attribute)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert payload["n_types"] == 2
    assert payload["n"] == n
