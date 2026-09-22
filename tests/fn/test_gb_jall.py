"""Tests for gb_jall.gibbons_joint_all_order."""

from morie.fn import _array_core as np

from morie.fn.gb_jall import gibbons_joint_all_order


def test_gb_jall_basic():
    """Test basic functionality."""
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # pdf must be callable; use a simple parent density
    f = lambda v: float(np.exp(-v**2 / 2) / np.sqrt(2 * np.pi))
    result = gibbons_joint_all_order(x, f)
    assert isinstance(result, dict)
    # Documented keys
    for key in ("pdf", "coef", "prod", "ordered", "n", "method"):
        assert key in result
    assert result["n"] == 5
    # coef is n! = 120
    assert result["coef"] == float(np.math_factorial(5)) if hasattr(np, "math_factorial") else 120.0
    # ordered should be 1 (int) since x is strictly increasing
    assert result["ordered"] == 1
    # Compute expected pdf independently from the formula
    import math
    expected_coef = float(math.factorial(5))
    expected_prod = 1.0
    for v in x:
        expected_prod *= f(float(v))
    expected_pdf = expected_coef * expected_prod
    assert abs(result["pdf"] - expected_pdf) < 1e-12
    assert abs(result["prod"] - expected_prod) < 1e-12


def test_gb_jall_edge():
    """Test edge cases."""
    # Test with x not strictly ordered -> pdf should be 0
    x = np.array([0.5, 0.3, 0.7])
    f = lambda v: float(np.exp(-v**2 / 2) / np.sqrt(2 * np.pi))
    result = gibbons_joint_all_order(x, f)
    assert isinstance(result, dict)
    assert result["ordered"] == 0
    assert result["pdf"] == 0.0
    assert result["n"] == 3

    # Test with single element (n=1)
    x = np.array([0.42])
    result = gibbons_joint_all_order(x, f)
    assert result["n"] == 1
    assert result["coef"] == 1.0
    assert result["ordered"] == 1
    expected_pdf_single = 1.0 * f(0.42)
    assert abs(result["pdf"] - expected_pdf_single) < 1e-12
