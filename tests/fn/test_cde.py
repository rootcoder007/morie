"""Tests for cde.controlled_direct_effect."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cde import controlled_direct_effect


def _design_matrix(X, M):
    """Build the OLS design matrix [1, X, M, X*M] from 1-D inputs."""
    n = len(X)
    return [[1.0, float(X[i]), float(M[i]), float(X[i]) * float(M[i])]
            for i in range(n)]


def _ols(D, y):
    """Plain-python OLS for the 4-parameter design used by CDE."""
    n = len(y)
    p = len(D[0])
    XtX = [[sum(D[i][r] * D[i][c] for i in range(n)) for c in range(p)]
           for r in range(p)]
    Xty = [sum(D[i][j] * y[i] for i in range(n)) for j in range(p)]
    inv = _inv4(XtX)
    return [sum(inv[j][k] * Xty[k] for k in range(p)) for j in range(p)]


def _inv4(A):
    """Invert a 4x4 symmetric positive-definite matrix by cofactors."""
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    M = [[A[1][1], A[1][2], A[1][3]],
         [A[2][1], A[2][2], A[2][3]],
         [A[3][1], A[3][2], A[3][3]]]
    detA = (A[0][0] * det3(M)
            - A[0][1] * det3([[A[1][0], A[1][2], A[1][3]],
                              [A[2][0], A[2][2], A[2][3]],
                              [A[3][0], A[3][2], A[3][3]]])
            + A[0][2] * det3([[A[1][0], A[1][1], A[1][3]],
                              [A[2][0], A[2][1], A[2][3]],
                              [A[3][0], A[3][1], A[3][3]]])
            - A[0][3] * det3([[A[1][0], A[1][1], A[1][2]],
                              [A[2][0], A[2][1], A[2][2]],
                              [A[3][0], A[3][1], A[3][2]]]))
    C = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            sub = [[A[i][j] for j in range(4) if j != c]
                   for i in range(4) if i != r]
            sign = -1.0 if (r + c) % 2 else 1.0
            C[r][c] = sign * (sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
                              - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
                              + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0]))
    return [[C[c][r] / detA for c in range(4)] for r in range(4)]


def test_cde_basic():
    """Test basic functionality with the documented (n,) shapes."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_m = np.random.default_rng(44)
    n = 100
    Y = rng_y.normal(0, 1, n)
    X = rng_x.normal(0, 1, n)
    M = rng_m.normal(0, 1, n)
    m = 0.5
    result = controlled_direct_effect(Y, X, M, m)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "cde" in result
    assert "intercept" in result
    assert "beta_x" in result
    assert "beta_m" in result
    assert "interaction" in result
    assert "se" in result
    assert "n" in result
    assert result["n"] == n

    # Independent recomputation of the CDE from the documented formula.
    xs = [float(v) for v in X]
    ys = [float(v) for v in Y]
    ms = [float(v) for v in M]
    D = _design_matrix(xs, ms)
    b = _ols(D, ys)
    expected_cde = b[1] + b[3] * float(m)
    assert abs(result["estimate"] - expected_cde) < 1e-8
    assert abs(result["cde"] - expected_cde) < 1e-8
    assert abs(result["beta_x"] - b[1]) < 1e-8
    assert abs(result["beta_m"] - b[2]) < 1e-8
    assert abs(result["interaction"] - b[3]) < 1e-8
    assert abs(result["intercept"] - b[0]) < 1e-8
    assert result["m"] == float(m)


def test_cde_zero_interaction():
    """With no interaction term the CDE equals beta_x at any m."""
    rng = np.random.default_rng(0)
    n = 60
    X = rng.integers(0, 2, n).astype(float)
    M = rng.normal(0, 1, n)
    # Y = 0.7 + 1.3*X + 0.4*M + noise -- interaction == 0 in population
    Y = 0.7 + 1.3 * X + 0.4 * M + rng.normal(0, 0.1, n)
    m = 1.25
    result = controlled_direct_effect(Y, X, M, m)
    # estimate and cde must be identical and match the same formula
    assert abs(result["estimate"] - result["cde"]) < 1e-12
    # At any m the CDE collapses to beta_x; re-derive beta_x independently.
    xs = [float(v) for v in X]
    ys = [float(v) for v in Y]
    ms = [float(v) for v in M]
    D = _design_matrix(xs, ms)
    b = _ols(D, ys)
    assert abs(result["beta_x"] - b[1]) < 1e-8
    assert abs(result["interaction"] - b[3]) < 1e-8
    expected = b[1] + b[3] * float(m)
    assert abs(result["estimate"] - expected) < 1e-8
    # With negligible interaction, CDE should be close to beta_x at this m.
    assert abs(result["estimate"] - b[1]) < 0.1


def test_cde_edge():
    """Edge case: minimum number of observations (n == 5, just enough)."""
    rng = np.random.default_rng(7)
    X = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    M = np.array([0.5, 1.5, -0.5, 2.0, 0.0])
    Y = np.array([0.1, 1.2, -0.3, 2.1, 0.9])
    m = -1.0
    result = controlled_direct_effect(Y, X, M, m)
    assert isinstance(result, dict)
    assert result["n"] == 5
    assert result["m"] == -1.0

    xs = [float(v) for v in X]
    ys = [float(v) for v in Y]
    ms = [float(v) for v in M]
    D = _design_matrix(xs, ms)
    b = _ols(D, ys)
    expected = b[1] + b[3] * float(m)
    assert abs(result["estimate"] - expected) < 1e-8
