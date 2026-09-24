"""Tests for gpdkl.deep_kernel_gp."""

import math

from morie.fn import _array_core as np

from morie.fn.gpdkl import deep_kernel_gp


def _to_list(a):
    """Convert a morie array to nested lists suitable for the function."""
    arr = np.asarray(a)
    return arr.tolist()


def test_gpdkl_basic():
    """Test basic functionality.

    With nn=None the feature map is the identity, so deep_kernel_gp reduces to
    a plain RBF GP. For a single test point the returned ``estimate`` is the
    posterior mean, given by k(x*, X) (K + s^2 I)^-1 y.
    """
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n = 20
    d = 3
    X = _to_list(rng_X.normal(0.0, 1.0, (n, d)))
    y = _to_list(rng_y.normal(0.0, 1.0, n))
    X_test = _to_list(rng_t.normal(0.0, 1.0, (5, d)))

    result = deep_kernel_gp(X, y, X_test, None,
                            lengthscale=1.0, variance=1.0, noise=0.01)

    # The implementation returns a RichResult; allow either dict-like access
    # or attribute access for the keys we check.
    def get(key):
        if hasattr(result, "payload"):
            return result.payload.get(key)
        if isinstance(result, dict):
            return result.get(key)
        return None

    assert get("estimate") is not None
    assert isinstance(get("mean"), list)
    assert isinstance(get("variance"), list)
    assert len(get("mean")) == 5
    assert len(get("variance")) == 5
    assert get("n") == n
    assert get("features") == d

    # Independent recomputation of the posterior mean for the first test point.
    ell = 1.0
    var = 1.0
    s2 = 0.01

    def sqdist(a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(len(a)))

    K = [[var * np.exp(-0.5 * sqdist(X[i], X[j]) / (ell * ell))
          for j in range(n)] for i in range(n)]
    for i in range(n):
        K[i][i] += s2

    # Cholesky solve via a small explicit routine using only numpy primitives.
    def matvec(M, v):
        return [sum(M[i][j] * v[j] for j in range(len(v)))
                for i in range(len(M))]

    def transpose(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    def cholesky(M):
        n_ = len(M)
        L = [[0.0] * n_ for _ in range(n_)]
        for i in range(n_):
            for j in range(i + 1):
                s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = np.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
        return L

    def solve_chol(L, v):
        n_ = len(L)
        # forward
        z = [0.0] * n_
        for i in range(n_):
            z[i] = (v[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
        # backward
        x = [0.0] * n_
        for i in range(n_ - 1, -1, -1):
            x[i] = (z[i] - sum(L[k][i] * x[k] for k in range(i + 1, n_))) / L[i][i]
        return x

    L = cholesky(K)
    alpha = solve_chol(L, y)

    x_star = X_test[0]
    kstar = [var * np.exp(-0.5 * sqdist(x_star, X[i]) / (ell * ell))
             for i in range(n)]
    expected_mean = sum(kstar[i] * alpha[i] for i in range(n))

    assert np.allclose(get("mean")[0], expected_mean, atol=1e-8)


def test_gpdkl_edge():
    """Test edge cases: with nn=None the function should still run."""
    rng_X = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    rng_t = np.random.default_rng(44)
    n = 15
    d = 2
    X = _to_list(rng_X.normal(0.0, 1.0, (n, d)))
    y = _to_list(rng_y.normal(0.0, 1.0, n))
    X_test = _to_list(rng_t.normal(0.0, 1.0, (4, d)))

    result = deep_kernel_gp(X, y, X_test, None,
                            lengthscale=2.0, variance=0.5, noise=0.1)

    mean = result["mean"]
    variance = result["variance"]
    assert len(mean) == 4
    assert len(variance) == 4
    assert all(math.isfinite(v) for v in mean)
    # A GP posterior variance is positive and, with noise s2 = 0.1 folded
    # into the prior variance 0.5, cannot exceed the prior.
    assert all(0.0 < v <= 0.5 + 1e-9 for v in variance)
    assert result["n"] == n
    assert result["features"] == d
