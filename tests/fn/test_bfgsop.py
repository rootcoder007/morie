"""Tests for bfgsop.bfgs."""

from morie.fn import _array_core as np

from morie.fn.bfgsop import bfgs


def _build_inputs(p=5, seed=42):
    """Build valid (H, s, y) inputs of correct dimension with y's > 0."""
    rng = np.random.default_rng(seed)
    # s: step
    s = rng.normal(0.0, 1.0, p)
    # y: gradient change; construct so curvature y's > 0 holds
    y = rng.normal(0.0, 1.0, p)
    y = y + 2.0 * s  # ensure y . s > 0 with overwhelming probability
    # H: square current approximation (start from identity)
    H = rng.normal(0.0, 1.0, (p, p))
    return H, s, y


def test_bfgsop_basic():
    """Test basic functionality of the BFGS inverse-Hessian update."""
    H, s, y = _build_inputs(p=5, seed=42)
    result = bfgs(H, s, y, inverse=True)

    assert isinstance(result, dict)

    # Documented keys
    for key in ("M", "rho", "curvature", "secant", "p", "inverse"):
        assert key in result, f"missing key: {key}"

    # Inverse form should have curvature = y's
    ys = float(sum(float(y[i]) * float(s[i]) for i in range(len(s))))
    assert result["curvature"] == ys
    assert result["rho"] == 1.0 / ys
    assert result["inverse"] is True
    assert result["p"] == len(s)

    # Secant residual should be small (within roundoff tolerance).
    # We don't compare a hard-coded number; we recompute from the returned M.
    M = result["M"]
    sec = [sum(float(M[i][j]) * float(y[j]) for j in range(len(y)))
           for i in range(len(y))]
    gap = max(abs(sec[i] - float(s[i])) for i in range(len(s)))
    assert gap == result["secant"]
    assert gap < 1e-8

    # Independent re-computation of the inverse update from the formula
    #   H_new = (I - rho s y') H (I - rho y s') + rho s s'
    p = len(s)
    rho = 1.0 / ys
    H_arr = [[float(H[i][j]) for j in range(p)] for i in range(p)]
    s_arr = [float(s[i]) for i in range(p)]
    y_arr = [float(y[i]) for i in range(p)]

    # L = I - rho * outer(s, y)
    L = [[(1.0 if i == j else 0.0) - rho * s_arr[i] * y_arr[j]
          for j in range(p)] for i in range(p)]
    # R = I - rho * outer(y, s)
    R = [[(1.0 if i == j else 0.0) - rho * y_arr[i] * s_arr[j]
          for j in range(p)] for i in range(p)]
    # T = L H R
    LH = [[sum(L[i][k] * H_arr[k][j] for k in range(p))
           for j in range(p)] for i in range(p)]
    T = [[sum(LH[i][k] * R[k][j] for k in range(p))
          for j in range(p)] for i in range(p)]
    # N = T + rho * outer(s, s)
    expected = [[T[i][j] + rho * s_arr[i] * s_arr[j]
                 for j in range(p)] for i in range(p)]

    for i in range(p):
        for j in range(p):
            assert abs(float(M[i][j]) - expected[i][j]) < 1e-10


def test_bfgsop_edge():
    """Test the BFGS Hessian (B-form) update path."""
    H, s, y = _build_inputs(p=5, seed=7)
    result = bfgs(H, s, y, inverse=False)

    assert isinstance(result, dict)
    assert "M" in result
    assert "rho" in result
    assert "curvature" in result
    assert "secant" in result
    assert "p" in result
    assert "inverse" in result
    assert result["inverse"] is False

    # Independent re-computation of the B-form update:
    #   B_new = B - B s s' B / (s' B s) + y y' / (y' s)
    p = len(s)
    B = [[float(H[i][j]) for j in range(p)] for i in range(p)]
    s_arr = [float(s[i]) for i in range(p)]
    y_arr = [float(y[i]) for i in range(p)]
    ys = sum(y_arr[i] * s_arr[i] for i in range(p))
    Bs = [sum(B[i][j] * s_arr[j] for j in range(p)) for i in range(p)]
    sBs = sum(s_arr[i] * Bs[i] for i in range(p))
    expected = [[B[i][j] - Bs[i] * Bs[j] / sBs + y_arr[i] * y_arr[j] / ys
                 for j in range(p)] for i in range(p)]
    M = result["M"]
    for i in range(p):
        for j in range(p):
            assert abs(float(M[i][j]) - expected[i][j]) < 1e-10
