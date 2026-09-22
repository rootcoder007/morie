"""Tests for assort.degree_assortativity."""

from morie.fn import _array_core as np

from morie.fn.assort import degree_assortativity


def test_assort_basic():
    """Test basic functionality."""
    n = 6
    A = np.zeros((n, n))
    # Build a simple undirected adjacency: star centered at 0
    # 0 connected to 1, 2, 3, 4, 5 -> center degree 5, leaves degree 1
    for j in range(1, n):
        A[0, j] = 1.0
        A[j, 0] = 1.0

    # y is ignored by the function (interface compatibility), but pass a vector
    y = np.random.default_rng(43).normal(0, 1, 100)

    result = degree_assortativity(y, A)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result

    # Independent computation of Newman's assortativity coefficient (excess degree)
    # Degrees from the adjacency
    deg = [0.0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0.0 or A[j, i] != 0.0:
                edges.append((i, j))
                deg[i] += 1.0
                deg[j] += 1.0
    M = len(edges)
    off = 1.0  # excess=True default
    sjk = 0.0
    sj = 0.0
    sk = 0.0
    sj2 = 0.0
    sk2 = 0.0
    for (u, v) in edges:
        for (p, q) in ((u, v), (v, u)):
            jj = deg[p] - off
            kk = deg[q] - off
            sjk += jj * kk
            sj += jj
            sk += kk
            sj2 += jj * jj
            sk2 += kk * kk
    m2 = 2.0 * M
    expected_r = (sjk - sj * sk / m2) / np.sqrt((sj2 - sj * sj / m2) * (sk2 - sk * sk / m2))

    assert np.isclose(result["r"], expected_r)
    assert result["M"] == M
    assert result["n"] == n
    assert list(result["degree"]) == deg
    assert result["excess"] is True
    assert result["method"].startswith("Newman")


def test_assort_edge():
    """Test edge cases."""
    n = 6
    A = np.zeros((n, n))
    for j in range(1, n):
        A[0, j] = 1.0
        A[j, 0] = 1.0

    y = np.random.default_rng(43).normal(0, 1, 100)

    result = degree_assortativity(y, A)
    assert isinstance(result, dict)
    assert "r" in result
    assert "M" in result
    assert "degree" in result
