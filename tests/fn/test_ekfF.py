"""Tests for ekfF.extended_kalman."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.ekfF import extended_kalman


def test_ekfF_basic():
    """Test basic functionality on a simple scalar+2D-state linear model."""
    rng = np.random.default_rng(43)
    d = 2
    n = 100

    # Observation sequence (scalars).
    y = rng.normal(0.0, 1.0, n)

    # Linear transition x_t = A x_{t-1} (constant map -> function of state).
    A = [[0.9, 0.1], [0.0, 0.8]]
    f = lambda x: [sum(A[i][k] * x[k] for k in range(d)) for i in range(d)]

    # Scalar observation map y_t = c' x_t.
    c = [0.3, 0.7]
    h = lambda x: sum(c[k] * x[k] for k in range(d))

    # Jacobians (constant matrices in this linear case).
    F = lambda x: [row[:] for row in A]
    H = lambda x: c[:]

    Q = [[0.1, 0.0], [0.0, 0.1]]
    R = 0.5  # must be a positive float

    result = extended_kalman(y, f, h, F, H, Q, R)

    assert isinstance(result, dict)

    # Documented keys.
    for key in ("estimate", "state", "cov", "loglik", "n", "method"):
        assert key in result

    assert result["n"] == n
    assert result["method"] == "Extended Kalman filter"
    assert len(result["state"]) == d
    assert len(result["cov"]) == d * d

    # First state component matches the first entry of the returned state.
    assert result["estimate"] == result["state"][0]

    # Log-likelihood must be finite.
    assert math.isfinite(result["loglik"])

    # Independent recomputation of the log-likelihood using the Joseph-style
    # scalar EKF recursion on the same data (same linear model, x0=zeros,
    # P0=I). This gives an independent reference for loglik that does not
    # call extended_kalman.
    x = [0.0] * d
    P = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
    Hk = c
    Fk = A
    ref_loglik = 0.0
    for t in range(n):
        xp = [sum(Fk[i][k] * x[k] for k in range(d)) for i in range(d)]
        FP = [[sum(Fk[i][k] * P[k][j] for k in range(d)) for j in range(d)]
              for i in range(d)]
        Pp = [[sum(FP[i][k] * Fk[j][k] for k in range(d)) + Q[i][j]
               for j in range(d)] for i in range(d)]
        PH = [sum(Pp[i][k] * Hk[k] for k in range(d)) for i in range(d)]
        S = sum(Hk[i] * PH[i] for i in range(d)) + R
        K = [PH[i] / S for i in range(d)]
        v = y[t] - sum(Hk[k] * xp[k] for k in range(d))
        x = [xp[i] + K[i] * v for i in range(d)]
        P = [[Pp[i][j] - K[i] * S * K[j] for j in range(d)] for i in range(d)]
        ref_loglik += -0.5 * (math.log(2.0 * math.pi * S) + v * v / S)

    assert math.isclose(result["loglik"], ref_loglik, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(result["state"][0], x[0], rel_tol=1e-10, abs_tol=1e-10)


def test_ekfF_edge():
    """Test edge cases: 1D state, constant matrices, and R must be positive float."""
    d = 1
    n = 50
    rng = np.random.default_rng(43)
    y = rng.normal(0.0, 1.0, n)

    # 1D linear model: x_t = 0.9 x_{t-1}, y_t = 0.3 x_t.
    a = 0.9
    c = 0.3
    f = lambda x: [a * x[0]]
    h = lambda x: c * x[0]
    F = lambda x: [[a]]
    H = lambda x: [c]

    Q = [[0.05]]
    R = 0.2  # positive float, as required

    result = extended_kalman(y, f, h, F, H, Q, R)

    assert isinstance(result, dict)
    for key in ("estimate", "state", "cov", "loglik", "n", "method"):
        assert key in result
    assert result["n"] == n
    assert len(result["state"]) == d
    assert len(result["cov"]) == d * d
    assert math.isfinite(result["loglik"])
