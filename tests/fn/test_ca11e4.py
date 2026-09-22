"""Tests for ca11e4.ca_chapter_11_equation_4."""

import math

from morie.fn import _array_core as np

from morie.fn.ca11e4 import ca_chapter_11_equation_4


def _independent_hedges_g(d, n1, n2):
    """Independent re-implementation of Hedges' g = J * d for verification."""
    df = n1 + n2 - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return J * d


def test_ca11e4_basic():
    """Test basic scalar functionality with documented arity."""
    d = 0.5
    n1 = 30
    n2 = 30
    result = ca_chapter_11_equation_4(d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result

    expected = _independent_hedges_g(d, n1, n2)
    assert math.isclose(result["value"], expected, rel_tol=1e-12)


def test_ca11e4_edge():
    """Test edge case with a larger effect and unequal group sizes."""
    d = 0.8
    n1 = 50
    n2 = 25
    result = ca_chapter_11_equation_4(d, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result

    expected = _independent_hedges_g(d, n1, n2)
    assert math.isclose(result["value"], expected, rel_tol=1e-12)
