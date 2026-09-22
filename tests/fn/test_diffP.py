"""Tests for diffP.diffpool."""

from morie.fn import _array_core as np

from morie.fn.diffP import diffpool


def _toy_adj():
    """Build a tiny 3x3 adjacency matrix as a plain Python list."""
    return [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]


def _toy_features():
    """Build a 3x2 feature matrix as a plain Python list."""
    return [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]


def _row_softmax(row):
    """Compute softmax of a list of floats, matching the implementation."""
    import math
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    z = sum(exps)
    return [v / z for v in exps]


def test_diffP_basic():
    """Test basic functionality."""
    A = _toy_adj()
    X = _toy_features()
    K_clusters = 2
    result = diffpool(A, X, K_clusters)
    assert isinstance(result, dict)

    # Documented return keys
    for key in ("estimate", "S", "A_pool", "H_pool",
                "link_loss", "entropy_loss", "n", "K"):
        assert key in result

    # Shapes
    n = len(A)
    K = K_clusters
    f = len(X[0])
    assert result["n"] == n
    assert result["K"] == K
    assert len(result["S"]) == n
    assert all(len(row) == K for row in result["S"])
    assert len(result["A_pool"]) == K
    assert all(len(row) == K for row in result["A_pool"])
    assert len(result["H_pool"]) == K
    assert all(len(row) == f for row in result["H_pool"])

    # Softmax rows sum to one
    import math
    for row in result["S"]:
        s = sum(row)
        assert math.isclose(s, 1.0, abs_tol=1e-9)

    # estimate equals link_loss
    assert math.isclose(result["estimate"], result["link_loss"], abs_tol=1e-12)

    # Independent computation of link_loss using the formula
    Sm = result["S"]
    ll = 0.0
    for i in range(n):
        for j in range(n):
            ss = sum(Sm[i][r] * Sm[j][r] for r in range(K))
            ll += (A[i][j] - ss) ** 2
    ll = math.sqrt(ll) / n
    assert math.isclose(result["link_loss"], ll, rel_tol=1e-9, abs_tol=1e-12)

    # Independent computation of A_pool: Ap[r][s] = sum_ij S[i,r] * A[i,j] * S[j,s]
    Ap = [[0.0] * K for _ in range(K)]
    for r in range(K):
        for s in range(K):
            acc = 0.0
            for i in range(n):
                for j in range(n):
                    acc += Sm[i][r] * A[i][j] * Sm[j][s]
            Ap[r][s] = acc
    for r in range(K):
        for s in range(K):
            assert math.isclose(result["A_pool"][r][s], Ap[r][s],
                                rel_tol=1e-9, abs_tol=1e-12)

    # Independent computation of H_pool
    Hp = [[sum(Sm[i][r] * X[i][t] for i in range(n)) for t in range(f)]
          for r in range(K)]
    for r in range(K):
        for t in range(f):
            assert math.isclose(result["H_pool"][r][t], Hp[r][t],
                                rel_tol=1e-9, abs_tol=1e-12)

    # Deterministic given the seed
    result2 = diffpool(A, X, K_clusters)
    for r in range(K):
        for s in range(K):
            assert math.isclose(result["A_pool"][r][s],
                                result2["A_pool"][r][s],
                                rel_tol=1e-12, abs_tol=1e-12)


def test_diffP_edge():
    """Test edge cases."""
    A = _toy_adj()
    X = _toy_features()
    K_clusters = 1
    result = diffpool(A, X, K_clusters)
    assert isinstance(result, dict)
    assert result["K"] == 1
    # With K=1, S has a single column of ones -> A_pool is sum of A entries,
    # H_pool is sum of X rows
    import math
    n = len(A)
    f = len(X[0])
    total = sum(sum(row) for row in A)
    assert math.isclose(result["A_pool"][0][0], total,
                        rel_tol=1e-9, abs_tol=1e-12)
    h_sum = [sum(X[i][t] for i in range(n)) for t in range(f)]
    for t in range(f):
        assert math.isclose(result["H_pool"][0][t], h_sum[t],
                            rel_tol=1e-9, abs_tol=1e-12)
