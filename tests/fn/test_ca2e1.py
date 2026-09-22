"""Tests for ca2e1.ca_chapter_2_equation_1."""

from morie.fn import _array_core as np

from morie.fn.ca2e1 import ca_chapter_2_equation_1


def test_ca2e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 2.5 + 0.75 * x + rng.normal(0, 0.1, 100)
    result = ca_chapter_2_equation_1(x, y)
    assert isinstance(result, dict)
    assert "b1" in result

    # Independent computation of the OLS slope via plain arithmetic.
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    expected_b1 = num / den

    assert abs(result["b1"] - expected_b1) < 1e-10


def test_ca2e1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 2.5 + 0.75 * x + rng.normal(0, 0.1, 100)
    result = ca_chapter_2_equation_1(x, y)
    assert isinstance(result, dict)
