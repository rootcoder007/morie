"""Tests for ca2e9.ca_chapter_2_equation_9."""

from morie.fn import _array_core as np

from morie.fn.ca2e9 import ca_chapter_2_equation_9


def test_ca2e9_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    true_b = 2.5
    x = rng.normal(0, 1, n)
    y = true_b * x + rng.normal(0, 1, n)
    # OLS estimate: b_hat = sum(x*y)/sum(x*x); se estimated from residuals
    b_hat = float(np.sum(x * y) / np.sum(x * x))
    resid = y - b_hat * x
    sigma2 = float(np.sum(resid * resid) / (n - 2))
    se_b = float(np.sqrt(sigma2 / np.sum(x * x)))
    result = ca_chapter_2_equation_9(b_hat, se_b)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation of the t-statistic from the documented formula
    expected_t = b_hat / se_b
    assert float(np.abs(float(result["value"]) - expected_t)) < 1e-10


def test_ca2e9_edge():
    """Test edge cases."""
    # Small, hand-checked values
    result = ca_chapter_2_equation_9(0.6, 0.2)
    assert isinstance(result, dict)
    assert "value" in result
    assert float(np.abs(float(result["value"]) - 3.0)) < 1e-12
    assert "method" in result
