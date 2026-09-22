"""Tests for gh_c6_12.ghosal_sep_consist."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c6_12 import ghosal_sep_consist


def test_gh_c6_12_basic():
    """Test basic functionality."""
    delta = 0.5
    k = 2
    n = 10
    result = ghosal_sep_consist(delta, k, n)
    assert isinstance(result, dict)
    assert "bound" in result
    assert "rate" in result
    assert "exponent" in result
    assert "delta" in result
    assert "k" in result
    assert "n" in result

    # Independent computation of the documented formula: bound = delta^{n/k}
    expected_bound = delta ** (n / k)
    expected_rate = -math.log(delta) / k
    expected_exponent = -expected_rate * n

    assert result["bound"] == math.exp(expected_exponent)
    assert math.isclose(result["bound"], expected_bound, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["rate"], expected_rate, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["exponent"], expected_exponent, rel_tol=1e-12, abs_tol=1e-12)
    assert result["delta"] == float(delta)
    assert result["k"] == float(k)
    assert result["n"] == float(n)


def test_gh_c6_12_edge():
    """Test edge cases."""
    # Smallest valid k and n
    delta = 0.25
    k = 1
    n = 1
    result = ghosal_sep_consist(delta, k, n)
    assert isinstance(result, dict)
    assert "bound" in result

    expected_bound = delta ** (n / k)
    assert math.isclose(result["bound"], expected_bound, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["delta"], float(delta), rel_tol=1e-12, abs_tol=1e-12)
