"""Tests for gb_jnt.gibbons_joint_order."""

import math

from morie.fn import _array_core as np

from morie.fn.gb_jnt import gibbons_joint_order


def _normal_cdf(z):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _normal_pdf(z):
    """Standard normal PDF."""
    return math.exp(-0.5 * float(z) ** 2) / math.sqrt(2.0 * math.pi)


def test_gb_jnt_basic():
    """Test basic functionality with valid scalar inputs and the
    documented formula from Gibbons & Chakraborti Sec. 2.5."""
    x_val = -0.5
    y_val = 0.7
    r = 2
    s = 4
    n = 6

    # Compute the joint pdf independently from the formula in the docstring
    # using plain arithmetic (no numpy on the expected value).
    Fx = _normal_cdf(x_val)
    Fy = _normal_cdf(y_val)
    fx = _normal_pdf(x_val)
    fy = _normal_pdf(y_val)
    coef = math.factorial(n) / (
        math.factorial(r - 1) * math.factorial(s - r - 1) * math.factorial(n - s)
    )
    expected_pdf = (
        coef
        * Fx ** (r - 1)
        * (Fy - Fx) ** (s - r - 1)
        * (1.0 - Fy) ** (n - s)
        * fx
        * fy
    )

    result = gibbons_joint_order(
        x_val, y_val, r, s, n, _normal_cdf, _normal_pdf
    )

    # The function returns a RichResult supporting dict-style access.
    assert "pdf" in result
    assert "coef" in result
    assert "fx" in result
    assert "fy" in result
    assert "r" in result and result["r"] == r
    assert "s" in result and result["s"] == s
    assert "n" in result and result["n"] == n
    assert "method" in result

    assert result["coef"] == coef
    assert math.isclose(result["pdf"], expected_pdf, rel_tol=1e-12, abs_tol=1e-15)
    assert result["pdf"] >= 0.0


def test_gb_jnt_edge():
    """Test that the joint pdf is zero when x >= y (outside the support)."""
    x_val = 0.5
    y_val = 0.3
    r = 2
    s = 4
    n = 6

    result = gibbons_joint_order(
        x_val, y_val, r, s, n, _normal_cdf, _normal_pdf
    )

    assert "pdf" in result
    assert result["pdf"] == 0.0
    # coef is still returned and is valid for the indices.
    expected_coef = math.factorial(n) / (
        math.factorial(r - 1) * math.factorial(s - r - 1) * math.factorial(n - s)
    )
    assert result["coef"] == expected_coef
