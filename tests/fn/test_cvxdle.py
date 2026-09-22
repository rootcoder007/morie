"""Tests for cvxdle.boyd_dual_norm."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.cvxdle import boyd_dual_norm


def test_cvxdle_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, 100)

    # Dual of l1 is l_inf: value = max |z|
    result = boyd_dual_norm(1, z)
    expected_value = float(np.max(np.abs(z)))
    assert isinstance(result, dict)
    assert result["value"] == expected_value
    assert result["dual_of"] == "inf"
    assert result["conjugate_exponent"] == float("inf")
    x = np.asarray(result["maximizer"], dtype=float).ravel()
    zflat = np.asarray(z, dtype=float).ravel()
    # The maximizer must lie in the primal l1 unit ball (max |x| <= 1)
    # and recover the supremum value via inner product with z.
    assert float(np.max(np.abs(x))) <= 1.0 + 1e-12
    assert abs(float(np.sum(zflat * x)) - expected_value) < 1e-10

    # Dual of l_inf is l1: value = sum |z|
    result_inf = boyd_dual_norm("inf", z)
    expected_l1 = float(np.sum(np.abs(z)))
    assert result_inf["value"] == expected_l1
    assert result_inf["dual_of"] == "1"
    assert result_inf["conjugate_exponent"] == 1.0
    x_inf = np.asarray(result_inf["maximizer"], dtype=float).ravel()
    # Maximizer for the l_inf dual lies in the l_inf unit ball (max |x| <= 1)
    assert float(np.max(np.abs(x_inf))) <= 1.0 + 1e-12
    assert abs(float(np.sum(zflat * x_inf)) - expected_l1) < 1e-10


def test_cvxdle_edge():
    """Test edge cases."""
    # l2 is self-dual: value = sqrt(sum(z**2))
    z2 = np.array([3.0, -4.0])
    result_l2 = boyd_dual_norm(2, z2)
    expected_l2 = float(np.sqrt(np.sum(z2 ** 2)))
    assert result_l2["value"] == expected_l2
    assert result_l2["dual_of"] == "2"
    assert result_l2["conjugate_exponent"] == 2.0
    x_l2 = np.asarray(result_l2["maximizer"], dtype=float).ravel()
    zflat = np.asarray(z2, dtype=float).ravel()
    # l2 maximizer lies in the l2 unit ball
    assert float(np.sqrt(np.sum(x_l2 ** 2))) <= 1.0 + 1e-12
    assert abs(float(np.sum(zflat * x_l2)) - expected_l2) < 1e-10

    # Frobenius norm on a matrix: self-dual, value = sqrt(sum(z**2))
    zmat = np.array([[3.0, -4.0], [0.0, 5.0]])
    result_fro = boyd_dual_norm("fro", zmat)
    expected_fro = float(np.sqrt(np.sum(zmat ** 2)))
    assert result_fro["value"] == expected_fro
    assert result_fro["dual_of"] == "fro"
    assert result_fro["conjugate_exponent"] == 2.0

    # docstring examples must hold exactly
    r1 = boyd_dual_norm(1, [3.0, -5.0, 2.0])
    assert r1["value"] == 5.0
    assert r1["dual_of"] == "inf"

    r2 = boyd_dual_norm("inf", [3.0, -5.0, 2.0])
    assert r2["value"] == 10.0

    r3 = boyd_dual_norm(2, [3.0, 4.0])
    assert round(r3["value"], 10) == 5.0
