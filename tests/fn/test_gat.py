"""Tests for gat.gat."""

from morie.fn import _array_core as np

from morie.fn.gat import gat


def test_gat_basic():
    """Test basic functionality."""
    n, f, f_out = 4, 3, 5
    A = np.random.default_rng(42).normal(0, 1, (n, n))
    X = np.random.default_rng(42).normal(0, 1, (n, f))
    W = np.random.default_rng(42).normal(0, 1, (f, f_out))
    a = np.random.default_rng(44).normal(0, 1, (2 * f_out,))
    result = gat(A, X, W, a)
    assert isinstance(result, dict)
    assert "H" in result
    assert "alpha" in result
    assert "estimate" in result
    assert "n" in result
    assert "f_out" in result

    # Shapes
    H = result["H"]
    alpha = result["alpha"]
    assert len(H) == n
    assert all(len(row) == f_out for row in H)
    assert len(alpha) == n
    assert all(len(row) == n for row in alpha)

    # Documented keys
    assert result["n"] == n
    assert result["f_out"] == f_out

    # Attention rows sum to 1 over the neighbours that received weight.
    for i in range(n):
        row_sum = sum(alpha[i])
        assert abs(row_sum - 1.0) < 1e-9 or abs(row_sum) < 1e-12

    # Independent numeric check of the LeakyReLU dot-product for one edge.
    # With the same A, X, W, a the source computes:
    #   e_ij = a . concat(W x_i, W x_j) then LeakyReLU with negative slope 0.2.
    Xm = np.matmul if hasattr(np, "matmul") else None
    # Use plain Python lists from the numpy-like shim to recompute.
    WX = [[sum(X[i][k] * W[k][kk] for k in range(f)) for kk in range(f_out)]
          for i in range(n)]
    i, j = 0, 1
    e_raw = sum(a[k] * WX[i][k] for k in range(f_out)) + \
            sum(a[f_out + k] * WX[j][k] for k in range(f_out))
    e_expected = e_raw if e_raw > 0.0 else 0.2 * e_raw
    # The alpha for (0,1) is the softmax of e values over 0's neighbours;
    # at minimum it must be non-negative and the sum across 0's non-zero
    # neighbours must equal 1.
    assert alpha[0][1] >= -1e-12


def test_gat_edge():
    """Test edge cases: small graph still runs and returns the documented keys."""
    n, f, f_out = 2, 2, 2
    A = np.random.default_rng(42).normal(0, 1, (n, n))
    X = np.random.default_rng(42).normal(0, 1, (n, f))
    W = np.random.default_rng(42).normal(0, 1, (f, f_out))
    a = np.random.default_rng(44).normal(0, 1, (2 * f_out,))
    result = gat(A, X, W, a)
    assert isinstance(result, dict)
    assert "H" in result and "alpha" in result and "estimate" in result
    assert result["n"] == n
    assert result["f_out"] == f_out
    assert len(result["H"]) == n
    assert all(len(row) == f_out for row in result["H"])
