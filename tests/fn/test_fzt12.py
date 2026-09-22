"""Tests for fzt12.fauzi_thm1_2_var_mgkde."""

from morie.fn import _array_core as np

from morie.fn.fzt12 import fauzi_thm1_2_var_mgkde


def test_fzt12_basic():
    """Test basic functionality: estimate matches J_h^2 / J_{4h}."""
    jh = 2.5
    j4h = 1.6
    result = fauzi_thm1_2_var_mgkde(jh, j4h)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "remainder" in result
    assert "t1" in result
    assert "t2" in result
    assert "method" in result
    # estimate = jh**2 / j4h, computed independently from the formula
    expected_estimate = jh * jh / j4h
    assert result["estimate"] == expected_estimate
    # Without (h, a, b, f) supplied, remainder is NaN per the docstring
    assert np.isnan(result["remainder"])
    # The trick exponents (2, -1) come out of the cancellation described
    assert result["t1"] == 2.0
    assert result["t2"] == -1.0


def test_fzt12_edge():
    """Test the explicit O(h) remainder when (h, a, b, f) are supplied."""
    jh = 2.5
    j4h = 1.6
    h = 0.3
    a = 0.4
    b = 0.7
    f = 1.2
    result = fauzi_thm1_2_var_mgkde(jh, j4h, h=h, a=a, b=b, f=f)
    assert isinstance(result, dict)
    expected_estimate = jh * jh / j4h
    assert result["estimate"] == expected_estimate
    # Explicit O(h) remainder: -2 * (b - a^2 / (2 f)) * h
    expected_rem = -2.0 * (b - a * a / (2.0 * f)) * h
    assert result["remainder"] == expected_rem
