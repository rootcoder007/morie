"""Tests for gpreg.gaussian_process_regression."""

from morie.fn import _array_core as np

from morie.fn.gpreg import gaussian_process_regression


def _se_kernel(u):
    """Squared-exponential kernel callable (signature k(x1, x2))."""
    return lambda x1, x2: np.exp(-0.5 * np.sum((np.asarray(x1) - np.asarray(x2)) ** 2))


def test_gpreg_basic():
    """Test basic functionality against the R&W (2006) eq. (2.23)-(2.24) formulas."""
    rng = np.random.default_rng(42)
    # Use small, hand-checkable shapes: n=4 training points, m=2 test points, d=2.
    X = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    y = [0.5, -1.2, 0.3, 1.1]
    X_test = [[0.5, 0.5], [-0.5, -0.5]]
    kernel = (1.0, 1.0)        # (sf, l) for the SE kernel
    noise = 0.1               # scalar noise std -> sn2 = 0.01

    result = gaussian_process_regression(X, y, X_test, kernel, noise)

    # Check shape and key names documented by the function.
    assert isinstance(result, dict)
    assert set(result.keys()) >= {"estimate", "variance", "loglik", "n", "method"}
    assert result["n"] == 4
    assert len(result["estimate"]) == 2
    assert len(result["variance"]) == 2

    # Independent reference computation of the predictive mean and variance.
    sf2, l2 = 1.0, 1.0
    sn2 = 0.01
    n = 4

    def k(xa, xb):
        dx = xa[0] - xb[0]
        dy = xa[1] - xb[1]
        return sf2 * np.exp(-0.5 * (dx * dx + dy * dy) / l2)

    K = [[k(X[i], X[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    Ks = [[k(X_test[p], X[i]) for i in range(n)] for p in range(2)]

    # Solve K @ alpha = y by explicit 4x4 inversion (independent of the function's solver).
    def inv4(M):
        # 4x4 matrix inverse via cofactors.
        def det3(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        out = [[0.0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                sub = [[M[i][j] for j in range(4) if j != c] for i in range(4) if i != r]
                co = ((-1) ** (r + c)) * det3(sub)
                out[c][r] = co  # transpose for inverse
        d = det3([[M[i][j] for j in range(3)] for i in range(3)])
        # full 4x4 det:
        def det4(m):
            s = 0.0
            for j in range(4):
                sub = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                s += ((-1) ** j) * m[0][j] * det3(sub)
            return s
        D = det4(M)
        return [[out[i][j] / D for j in range(4)] for i in range(4)]

    Kinv = inv4(K)

    def dot(u, v):
        return sum(u[i] * v[i] for i in range(len(u)))

    # alpha = K^{-1} y
    alpha = [dot(Kinv[i], y) for i in range(n)]
    # mean_p = Ks_p . alpha
    exp_mean = [dot(Ks[p], alpha) for p in range(2)]
    # var_p = k(x*,x*) - Ks_p . K^{-1} . Ks_p^T
    exp_var = []
    for p in range(2):
        v = [dot(Kinv[i], Ks[p]) for i in range(n)]
        exp_var.append(k(X_test[p], X_test[p]) - dot(Ks[p], v))

    # Compare element-wise to the function's output.
    for p in range(2):
        assert np.allclose(result["estimate"][p], exp_mean[p], atol=1e-10)
        assert np.allclose(result["variance"][p], exp_var[p], atol=1e-10)

    # Log marginal likelihood per eq. (2.30): -1/2 y^T K^{-1} y - 1/2 log|K| - n/2 log(2 pi).
    sign, logabsdet = np.linalg.slogdet(np.asarray(K))
    # Compute log determinant directly with the function's helper semantics:
    # we trust np.linalg above only for the numeric reference.
    exp_loglik = (
        -0.5 * sum(y[i] * alpha[i] for i in range(n))
        - 0.5 * logabsdet
        - 0.5 * n * float(np.log(2.0 * np.pi))
    )
    assert np.allclose(result["loglik"], exp_loglik, atol=1e-10)


def test_gpreg_edge():
    """Test edge cases: zero noise, custom callable kernel, 1-D inputs."""
    # Zero noise (default), custom callable kernel, 1-D inputs n=3, m=1, d=1.
    X = [[0.0], [1.0], [2.0]]
    y = [1.0, 2.0, 3.0]                # roughly linear in x
    X_test = [[1.5]]
    kernel = _se_kernel(np)            # callable: k(x1, x2) = exp(-0.5 ||x1-x2||^2)
    result = gaussian_process_regression(X, y, X_test, kernel, 0.0)

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"estimate", "variance", "loglik", "n", "method"}
    assert result["n"] == 3
    assert len(result["estimate"]) == 1
    assert len(result["variance"]) == 1

    # n=3 training points, m=1 test point -> a 3x3 linear system; invert by hand.
    sf2, l2 = 1.0, 1.0     # the callable above hardcodes these
    sn2 = 0.0
    n = 3

    def k(xa, xb):
        return sf2 * np.exp(-0.5 * (xa[0] - xb[0]) ** 2 / l2)

    K = [[k(X[i], X[j]) + (sn2 if i == j else 0.0) for j in range(n)] for i in range(n)]
    Ks = [[k(X_test[0], X[i]) for i in range(n)]]

    def inv3(m):
        d = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        inv = [[0.0] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                sub = [[m[i][j] for j in range(3) if j != c] for i in range(3) if i != r]
                co = ((-1) ** (r + c)) * (sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0])
                inv[c][r] = co / d
        return inv

    Kinv = inv3(K)

    def dot(u, v):
        return sum(u[i] * v[i] for i in range(len(u)))

    alpha = [dot(Kinv[i], y) for i in range(n)]
    exp_mean = [dot(Ks[p], alpha) for p in range(1)]
    v = [dot(Kinv[i], Ks[0]) for i in range(n)]
    exp_var = [k(X_test[0], X_test[0]) - dot(Ks[0], v)]

    assert np.allclose(result["estimate"][0], exp_mean[0], atol=1e-10)
    assert np.allclose(result["variance"][0], exp_var[0], atol=1e-10)
    # Variance must be non-negative.
    assert result["variance"][0] >= -1e-12
