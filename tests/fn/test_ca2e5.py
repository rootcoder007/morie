"""Tests for ca2e5.ca_chapter_2_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca2e5 import ca_chapter_2_equation_5


def test_ca2e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = 2.5 * x + rng.normal(0, 0.5, 100)
    result = ca_chapter_2_equation_5(x, y)

    # Function returns a RichResult (dict subclass)
    assert isinstance(result, dict)

    # Independent computation of the documented formula:
    # t = b1 / sqrt((sum((y - yhat)^2) / (n - 2)) / sum((x - xbar)^2))
    n = len(x)
    xbar = x.mean()
    ybar = y.mean()
    b1 = ((x - xbar) * (y - ybar)).sum() / ((x - xbar) ** 2).sum()
    b0 = ybar - b1 * xbar
    yhat = b0 + b1 * x
    rss = ((y - yhat) ** 2).sum()
    s2 = rss / (n - 2)
    sxx = ((x - xbar) ** 2).sum()
    expected_t = b1 / np.sqrt(s2 / sxx)

    # Headline key is 't' per the docstring
    assert "t" in result
    assert result["t"] == expected_t

    # Method should be set to the documented citation
    assert result["method"] == "Weisburd et al. (2022) eq. (2.5)"


def test_ca2e5_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 1, 100)  # unrelated y -> t should be roughly O(1)
    result = ca_chapter_2_equation_5(x, y)
    assert isinstance(result, dict)
    assert "t" in result
    assert abs(result["t"]) < 5.0  # extremely unlikely to be larger
