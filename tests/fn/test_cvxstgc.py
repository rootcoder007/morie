"""Tests for cvxstgc.boyd_strong_convex."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cvxstgc import boyd_strong_convex


def test_cvxstgc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    # Quadratic f(z) = 0.5 * z^T Q z with Q = diag(2, 5).
    # Its strong-convexity modulus is m = lambda_min(Q) = 2.
    Q = np.diag([2.0, 5.0])
    f = lambda z: 0.5 * float(z @ Q @ z)
    gf = lambda z: Q @ z

    x = np.array([1.0, 1.0])
    m = 2.0  # exact modulus -> inequality holds for all y

    # Probe points near x.
    ys = x + rng.uniform(-1.0, 1.0, (64, 2))
    ys = np.vstack([ys, x + np.eye(2), x - np.eye(2)])

    result = boyd_strong_convex(f, gf, x, m, y_samples=ys)

    # Independent recomputation of gaps from the documented formula.
    fx = float(f(x))
    gx = gf(x)
    d = ys - x
    lhs = np.array([float(f(y)) for y in ys])
    rhs = fx + (gx @ d.T).ravel() + 0.5 * m * np.sum(d * d, axis=1)
    expected_min_gap = float((lhs - rhs).min())
    expected_viol = int(np.sum(lhs - rhs < -1e-9))
    expected_bound = float(gx @ gx) / (2.0 * m)

    assert isinstance(result, dict)
    assert "holds" in result
    assert "violations" in result
    assert "worst_gap" in result
    assert "suboptimality_bound" in result
    assert "m" in result

    assert result["m"] == m
    assert result["violations"] == expected_viol
    assert result["worst_gap"] == expected_min_gap
    assert result["suboptimality_bound"] == expected_bound
    assert result["holds"] is True


def test_cvxstgc_edge():
    """Test edge cases."""
    Q = np.diag([2.0, 5.0])
    f = lambda z: 0.5 * float(z @ Q @ z)
    gf = lambda z: Q @ z

    x = np.array([1.0, 1.0])
    m_too_large = 5.5  # exceeds lambda_min(Q)=2 -> inequality fails somewhere
    ys = np.array([[1.0, 1.0],
                   [1.0 + 1.0, 1.0],
                   [1.0, 1.0 + 1.0]])

    result = boyd_strong_convex(f, gf, x, m_too_large, y_samples=ys)

    assert isinstance(result, dict)
    assert result["m"] == m_too_large
    assert result["holds"] is False
    assert result["violations"] >= 1

    # Independent recomputation of expected violation count.
    fx = float(f(x))
    gx = gf(x)
    d = ys - x
    lhs = np.array([float(f(y)) for y in ys])
    rhs = fx + (gx @ d.T).ravel() + 0.5 * m_too_large * np.sum(d * d, axis=1)
    expected_viol = int(np.sum(lhs - rhs < -1e-9))
    assert result["violations"] == expected_viol
