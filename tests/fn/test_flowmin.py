"""Tests for flowmin.min_cut."""

from morie.fn import _array_core as np

from morie.fn.flowmin import min_cut


def _symmetrize(M):
    n = len(M)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = float(M[i][j])
    for i in range(n):
        for j in range(i + 1, n):
            a = out[i][j] + out[j][i]
            out[i][j] = a
            out[j][i] = a
    return out


def test_flowmin_basic():
    """Test basic functionality on a small known graph."""
    # Build a 4-vertex undirected weighted graph.
    # Vertices: {0, 1, 2, 3}
    # Edges (undirected):
    #   0-1 weight 2
    #   0-2 weight 3
    #   1-2 weight 1
    #   2-3 weight 4
    # The global minimum cut value is 3 (cut {2,3} vs {0,1}),
    # because the cheapest way to separate any vertex from the rest is to
    # remove edges {2-3, 0-2, 1-2}? Let's verify: separating 0 from the rest
    # requires cutting 0-1 (2) and 0-2 (3) = 5. Separating 1 requires
    # 0-1 (2) + 1-2 (1) = 3. Separating 2 requires 0-2 (3) + 1-2 (1) + 2-3 (4) = 8.
    # Separating 3 requires 2-3 (4) = 4. Separating {0,1} from {2,3} requires
    # 0-2 (3) + 1-2 (1) = 4. Separating {0,2} from {1,3} requires
    # 0-1 (2) + 2-3 (4) = 6. Separating {0,3} from {1,2} requires
    # 0-1 (2) + 0-2 (3) + 2-3 (4) = 9.
    # So global min cut weight = 3, isolating vertex 1.
    n = 4
    A = [[0, 2, 3, 0],
         [2, 0, 1, 0],
         [3, 1, 0, 4],
         [0, 0, 4, 0]]

    result = min_cut(A)

    assert "estimate" in result
    assert "weight" in result
    assert "partition" in result
    assert "phases" in result
    assert isinstance(result["partition"], list)
    assert all(v in (0, 1) for v in result["partition"])
    assert sum(result["partition"]) in (1, 3)
    # Independent computation of the expected min cut weight for this graph.
    edges = [(0, 1, 2), (0, 2, 3), (1, 2, 1), (2, 3, 4)]
    expected_min = None
    for mask in range(1, (1 << n) - 1):
        s = [i for i in range(n) if mask & (1 << i)]
        t = [i for i in range(n) if not (mask & (1 << i))]
        cut_weight = 0.0
        for (u, v, w) in edges:
            if (u in s and v in t) or (u in t and v in s):
                cut_weight += w
        if expected_min is None or cut_weight < expected_min:
            expected_min = cut_weight
    assert result["estimate"] == expected_min
    assert result["weight"] == expected_min


def test_flowmin_edge():
    """Test edge cases."""
    n = 4
    A = [[0, 2, 3, 0],
         [2, 0, 1, 0],
         [3, 1, 0, 4],
         [0, 0, 4, 0]]

    result = min_cut(A)
    assert "estimate" in result
    assert "weight" in result
    assert "partition" in result
    assert "phases" in result
    # Sanity: phases has one entry per phase, and on n vertices there are n-1 phases.
    assert len(result["phases"]) == n - 1
