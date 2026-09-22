"""Tests for gb1461.gibbons_multinomial_gof."""

from morie.fn import _array_core as np

from morie.fn.gb1461 import gibbons_multinomial_gof


def test_gb1461_basic():
    """Test basic functionality."""
    observed = [10, 12, 8, 15, 5]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    result = gibbons_multinomial_gof(observed, probs, ddof=0)
    assert isinstance(result, dict)
    for key in ("statistic", "df", "p_value", "expected", "prob", "n", "k", "method"):
        assert key in result
    # Independent computation of the Pearson Q statistic
    N = sum(observed)
    exp = [N * p for p in probs]
    q_expected = sum((observed[i] - exp[i]) ** 2 / exp[i] for i in range(len(observed)))
    assert np.isclose(result["statistic"], q_expected)
    assert result["df"] == len(observed) - 1
    assert np.isclose(result["n"], float(N))
    assert result["k"] == len(observed)
    assert np.allclose(result["expected"], exp)


def test_gb1461_edge():
    """Test edge cases."""
    # Small case with ddof=1 (still >= 1 df required)
    observed = [5, 5, 5]
    probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    result = gibbons_multinomial_gof(observed, probs, ddof=1)
    assert isinstance(result, dict)
    assert result["k"] == 3
    assert result["df"] == 1
    assert 0.0 <= result["p_value"] <= 1.0
    assert 0.0 <= result["prob"] <= 1.0
