"""Tests for fzt23.fauzi_thm2_3_var_brdkdfe."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.fzt23 import fauzi_thm2_3_var_brdkdfe


def test_fzt23_basic():
    """Test basic functionality."""
    n = 100
    h = 0.3
    a = 0.5
    fx = 0.5
    density = 0.4
    r1 = 1.0
    r2 = 1.0
    result = fauzi_thm2_3_var_brdkdfe(n, h, a, fx, density, r1=r1, r2=r2)
    assert isinstance(result, dict)
    assert "variance" in result
    assert "se" in result
    assert "edfvar" in result
    assert "gain" in result
    assert "r1" in result
    assert "r2" in result
    assert "method" in result

    # Independent computation of the documented formula:
    # Var = f(1-f)/n - (h/n) * [ 2(a^4+1)/(a^2-1)^2 * r1 + r2 ] * density
    edfvar = fx * (1.0 - fx) / n
    bracket = 2.0 * (a ** 4 + 1.0) / (a * a - 1.0) ** 2 * r1 + r2
    expected_var = edfvar - h / n * bracket * density
    assert result["variance"] == expected_var
    assert result["edfvar"] == edfvar
    assert result["gain"] == edfvar - expected_var
    assert result["r1"] == r1
    assert result["r2"] == r2


def test_fzt23_edge():
    """Test edge cases."""
    n = 100
    h = 0.3
    a = 0.5
    fx = 0.5
    density = 0.4
    r1 = 1.0
    r2 = 1.0
    result = fauzi_thm2_3_var_brdkdfe(n, h, a, fx, density, r1=r1, r2=r2)
    assert isinstance(result, dict)
    assert result["method"] == "variance of the bias-reduced KDFE (Theorem 2.3)"
