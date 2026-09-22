"""Tests for ca2e2.ca_chapter_2_equation_2."""

from morie.fn import _array_core as np

from morie.fn.ca2e2 import ca_chapter_2_equation_2


def test_ca2e2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_2(x, y)
    assert isinstance(result, dict)
    # The function documents 'b1' as the headline key for the OLS slope.
    assert "b1" in result

    # Independent computation of the literature formula:
    # b1 = sum((xi - xbar)(yi - ybar)) / sum((xi - xbar)^2)
    xbar = np.mean(x)
    ybar = np.mean(y)
    num = np.sum((x - xbar) * (y - ybar))
    den = np.sum((x - xbar) ** 2)
    expected_b1 = float(num) / float(den)

    assert isinstance(result["b1"], float)
    assert result["b1"] == expected_b1


def test_ca2e2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Two distinct, non-constant inputs so the denominator is non-zero.
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_2(x, y)
    assert isinstance(result, dict)
    assert "b1" in result
    # The documented method label must be present in the payload.
    assert result.get("method") == "Weisburd et al. (2022) eq. (2.2)"
