"""Tests for blups.blup_random_slope."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.blups import blup_random_slope


def test_blups_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(43)
    rng_grp = np.random.default_rng(42)
    rng_D = np.random.default_rng(42)

    y = rng_y.normal(0, 1, 100)
    # X has 5 columns => fixed-effect design with 5 predictors
    X = rng_X.normal(0, 1, (100, 5))
    # Z has 10 columns => q = 10 random coefficients per group
    Z = rng_Z.normal(0, 1, (100, 10))
    # group must be integer-valued group labels per observation
    cluster = rng_grp.integers(0, 5, 100).astype(int).tolist()
    # D must be q by q => 10 by 10
    D = rng_D.normal(0, 1, (10, 10))
    sigma2_e = 1.5  # must be a strictly positive scalar

    beta = np.zeros(5).tolist()

    result = blup_random_slope(y, cluster, Z, D, sigma2_e, X=X, beta=beta)

    assert isinstance(result, dict)
    # The function returns a RichResult-like dict with documented keys
    assert "v" in result
    assert "levels" in result
    assert "nj" in result
    assert "fitted" in result
    assert "J" in result
    assert "q" in result
    assert "n" in result

    # Shape sanity checks derived from the documented formula and inputs
    assert result["n"] == 100
    assert result["q"] == 10
    assert len(result["levels"]) == result["J"]
    assert sum(result["nj"]) == 100
    assert len(result["fitted"]) == 100
    # one v vector (length q) per group
    assert len(result["v"]) == result["J"]
    for vj in result["v"]:
        assert len(vj) == 10

    # Independent verification of the BLUP formula for one group.
    # vhat_j = D Z_j' (Z_j D Z_j' + s2e I)^{-1} (y_j - X_j beta),
    # computed with plain arithmetic on the same inputs.
    levels = result["levels"]
    n = 100
    # Convert inputs to plain Python lists for the arithmetic
    y_list = y.tolist() if hasattr(y, "tolist") else list(y)
    Z_list = Z.tolist() if hasattr(Z, "tolist") else list(Z)
    X_list = X.tolist() if hasattr(X, "tolist") else list(X)
    D_list = D.tolist() if hasattr(D, "tolist") else list(D)

    # residual r = y - X beta (here beta = 0, so r = y)
    r_list = y_list

    def _matmul(A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        out = [[0.0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for k in range(cols_A):
                aik = A[i][k]
                for j in range(cols_B):
                    out[i][j] += aik * B[k][j]
        return out

    def _matinv(M):
        n = len(M)
        # build augmented matrix [M | I]
        aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i in range(n)]
        # Gauss-Jordan elimination
        for i in range(n):
            # find pivot
            piv = aug[i][i]
            assert piv != 0.0, "singular matrix in independent verification"
            for j in range(2 * n):
                aug[i][j] /= piv
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        return [row[n:] for row in aug]

    for L_idx, L in enumerate(levels):
        idx = [i for i in range(n) if cluster[i] == L]
        Zj = [Z_list[i] for i in idx]
        rj = [r_list[i] for i in idx]
        # Zj D Zj' is (m x m), add s2e I
        ZD = _matmul(Zj, D_list)            # (m x q)
        ZDZt = _matmul(ZD, [[Zj[r][c] for r in range(len(Zj))]
                            for c in range(len(Zj[0]))])  # (m x m)
        m = len(idx)
        M = [[ZDZt[a][b] + (sigma2_e if a == b else 0.0)
              for b in range(m)] for a in range(m)]
        Minv = _matinv(M)
        # w = M^{-1} r_j
        w = [sum(Minv[a][b] * rj[b] for b in range(m)) for a in range(m)]
        # vhat = D Z_j' w  =>  Zj' is (q x m), so D (q x q) @ Zj' (q x m) gives (q x m); then @ w (m,)
        Zjt = [[Zj[b][a] for b in range(m)] for a in range(10)]
        DZjt = _matmul(D_list, Zjt)
        vhat = [sum(DZjt[a][b] * w[b] for b in range(m)) for a in range(10)]
        assert len(vhat) == len(result["v"][L_idx])
        # Compare each component with a small tolerance
        for a in range(10):
            assert abs(vhat[a] - result["v"][L_idx][a]) < 1e-6


def test_blups_edge():
    """Test edge cases: a single group spanning all observations."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_Z = np.random.default_rng(43)
    rng_grp = np.random.default_rng(42)
    rng_D = np.random.default_rng(42)

    y = rng_y.normal(0, 1, 50)
    X = rng_X.normal(0, 1, (50, 3))
    Z = rng_Z.normal(0, 1, (50, 4))
    cluster = rng_grp.integers(0, 1, 50).astype(int).tolist()  # single group
    D = rng_D.normal(0, 1, (4, 4))
    sigma2_e = 0.7
    beta = np.zeros(3).tolist()

    result = blup_random_slope(y, cluster, Z, D, sigma2_e, X=X, beta=beta)

    assert isinstance(result, dict)
    assert "v" in result
    assert result["J"] == 1
    assert result["q"] == 4
    assert result["n"] == 50
    assert result["nj"] == [50]
    assert len(result["v"]) == 1
    assert len(result["v"][0]) == 4
    assert len(result["fitted"]) == 50
