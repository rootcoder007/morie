"""Tests for cvxlpl.boyd_linear_program_dual."""

from morie.fn import _array_core as np

from morie.fn.cvxlpl import boyd_linear_program_dual


def test_cvxlpl_basic():
    """Test basic functionality: shape correctness and documented keys."""
    rng = np.random.default_rng(42)
    m, n = 4, 7
    A = rng.normal(0, 1, (m, n))
    b = rng.normal(0, 1, m)  # length m matches A rows
    c = rng.normal(0, 1, n)  # length n matches A cols
    result = boyd_linear_program_dual(A, b, c)
    assert isinstance(result, dict)
    # Documented payload keys
    for key in ("y", "dual_value", "primal_value", "gap",
                "strong_duality", "shadow_prices"):
        assert key in result, f"missing documented key: {key}"


def test_cvxlpl_edge():
    """Test edge cases: strong duality and shadow-price interpretation."""
    # Simple, hand-crafted feasible LP so the answer is exactly checkable
    # from the formula in the docstring.
    A = np.array([[1.0, 1.0]])           # shape (1, 2)
    b = np.array([4.0])                   # shape (1,)
    c = np.array([1.0, 2.0])              # shape (2,)
    result = boyd_linear_program_dual(A, b, c)

    assert isinstance(result, dict)
    assert bool(result["strong_duality"]) is True

    # Independently computed expectations from the formula in the docstring.
    # min 1*x1 + 2*x2  s.t.  x1 + x2 == 4,  x1, x2 >= 0.
    # On the constraint x1 + x2 = 4, x2 = 4 - x1; objective = x1 + 2(4 - x1)
    # = 8 - x1, minimized at x1 = 4, x2 = 0, value = 4.
    expected_primal = 1.0 * 4.0 + 2.0 * 0.0        # == 4.0
    # Dual: max 4*y s.t. y <= 1, y <= 2  -> y* = 1, dual value = 4.
    expected_dual = b[0] * 1.0                     # b'y with y* = min(c)
    expected_gap = expected_primal - expected_dual

    assert abs(result["primal_value"] - expected_primal) < 1e-7
    assert abs(result["dual_value"] - expected_dual) < 1e-7
    assert abs(result["gap"] - expected_gap) < 1e-7

    # y is the shadow price; nudging b should move primal_value by y.
    y = float(np.asarray(result["y"]).ravel()[0])
    A2 = np.array([[1.0, 1.0]])
    b2 = np.array([5.0])
    c2 = np.array([1.0, 2.0])
    r2 = boyd_linear_program_dual(A2, b2, c2)
    delta_pv = r2["primal_value"] - result["primal_value"]
    assert abs(delta_pv - y) < 1e-7
