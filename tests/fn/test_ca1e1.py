"""Tests for ca1e1.ca_chapter_1_equation_1."""

from morie.fn import _array_core as np

from morie.fn.ca1e1 import ca_chapter_1_equation_1


def test_ca1e1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # b0 is the intercept (scalar), bs are coefficients, xs are predictor values
    # Formula: Yhat = b0 + sum(bk * xk)
    b0 = 1.5
    bs = rng.normal(0, 1, 5)
    xs = rng.normal(0, 1, 5)
    result = ca_chapter_1_equation_1(b0, bs, xs)
    
    # Independently compute expected value using the documented formula
    expected = b0 + np.sum(bs * xs)
    
    assert isinstance(result, dict)
    assert "value" in result
    # Verify the computed value matches the formula
    assert np.allclose(result["value"], expected)


def test_ca1e1_edge():
    """Test edge cases: all zeros."""
    b0 = 0.0
    bs = np.zeros(3)
    xs = np.zeros(3)
    result = ca_chapter_1_equation_1(b0, bs, xs)
    
    # Independent computation
    expected = 0.0 + np.sum(np.zeros(3) * np.zeros(3))
    
    assert isinstance(result, dict)
    assert "value" in result
    assert np.allclose(result["value"], expected)
