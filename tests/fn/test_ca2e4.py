"""Tests for ca2e4.ca_chapter_2_equation_4."""

from morie.fn import _array_core as np

from morie.fn.ca2e4 import ca_chapter_2_equation_4


def test_ca2e4_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_4(x, y)
    assert isinstance(result, dict)
    # Headline key per docstring is 'r' (Pearson correlation coefficient).
    assert "r" in result


def test_ca2e4_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)
    result = ca_chapter_2_equation_4(x, y)
    assert isinstance(result, dict)
    assert "r" in result

    # Independent computation of the documented formula
    # r = sum((yi-ybar)(xi-xbar)) / sqrt(sum(yi-ybar)^2 sum(xi-xbar)^2)
    xbar = np.mean(x)
    ybar = np.mean(y)
    num = np.sum((y - ybar) * (x - xbar))
    den = np.sqrt(np.sum((y - ybar) ** 2) * np.sum((x - xbar) ** 2))
    expected_r = num / den

    assert result["r"] == expected_r
    assert result["method"] == "Weisburd et al. (2022) eq. (2.4)"
    assert result["value"] == expected_r
