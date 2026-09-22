"""Tests for gpvarF.gp_variance."""

from morie.fn import _array_core as np

from morie.fn.gpvarF import gp_variance


def test_gpvarF_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (10, 3))
    X_star = rng.normal(0, 1, (5, 3))
    kernel = (1.0, 1.0)
    sigma2 = 0.1
    result = gp_variance(X, X_star, kernel, sigma2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "prior" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 10
    assert len(result["estimate"]) == 5
    assert len(result["prior"]) == 5
    # Prior variances k(x*, x*) for squared-exponential kernel with sf=1, l=1
    # are exactly 1.0
    for p in result["prior"]:
        assert abs(p - 1.0) < 1e-12
    # Estimate equals prior minus k* @ (K + sn^2 I)^{-1} @ k*.
    # Compute independently using plain arithmetic.
    sf, l = kernel
    sn2 = float(sigma2)
    n = 10
    m = 5
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sq = 0.0
            for d in range(3):
                diff = X[i][d] - X[j][d]
                sq += diff * diff
            K[i][j] = sf * sf * np.exp(-0.5 * sq / (l * l))
            if i == j:
                K[i][j] += sn2
    # Solve K w = k* for each test point using Cramer's rule for the
    # general n x n system via Gaussian elimination.
    def solve(A, b):
        N = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        # forward elimination
        for i in range(N):
            pivot = M[i][i]
            for r in range(i + 1, N):
                factor = M[r][i] / pivot
                for c in range(i, N + 1):
                    M[r][c] -= factor * M[i][c]
        # back substitution
        x = [0.0] * N
        for i in range(N - 1, -1, -1):
            s = M[i][N]
            for j in range(i + 1, N):
                s -= M[i][j] * x[j]
            x[i] = s / M[i][i]
        return x

    for p in range(m):
        ks = [0.0] * n
        sq_pp = 0.0
        for d in range(3):
            diff = X_star[p][d] - X_star[p][d]
            sq_pp += diff * diff
        kpp = sf * sf * np.exp(-0.5 * sq_pp / (l * l))
        for i in range(n):
            sq = 0.0
            for d in range(3):
                diff = X_star[p][d] - X[i][d]
                sq += diff * diff
            ks[i] = sf * sf * np.exp(-0.5 * sq / (l * l))
        w = solve(K, ks)
        expected = kpp - sum(ks[i] * w[i] for i in range(n))
        assert abs(result["estimate"][p] - expected) < 1e-9
        # Predictive variance must be non-negative
        assert result["estimate"][p] >= -1e-12


def test_gpvarF_edge():
    """Test edge cases: zero noise, single training point."""
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (4, 2))
    X_star = rng.normal(0, 1, (3, 2))
    kernel = (2.0, 1.5)
    sigma2 = 0.0
    result = gp_variance(X, X_star, kernel, sigma2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "prior" in result
    assert "n" in result
    assert result["n"] == 4
    assert len(result["estimate"]) == 3
    assert len(result["prior"]) == 3
    # With zero observation noise, the predictive variance for a training
    # point should be zero.
    for i, x in enumerate(X):
        idx = None
        for p in range(len(X_star)):
            match = True
            for d in range(2):
                if X_star[p][d] != x[d]:
                    match = False
                    break
            if match:
                idx = p
                break
        if idx is not None:
            assert abs(result["estimate"][idx]) < 1e-9
