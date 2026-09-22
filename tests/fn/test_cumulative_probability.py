"""Tests for cumulative_probability.cumulative_probability."""

from morie.fn import _array_core as np

from morie.fn.cumulative_probability import cumulative_probability


def test_ca5e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # probs: discrete probability distribution over y = 1..k, nonnegative and summing to 1
    raw = rng.random(100)
    probs = raw / raw.sum()
    result = cumulative_probability(probs, m=10)
    assert isinstance(result, dict)
    assert "value" in result
    # P(y <= m) computed independently from the formula: sum_{j=1..m} P(y=j)
    expected = float(np.sum(probs[:10]))
    assert abs(result["value"] - expected) < 1e-10


def test_ca5e6_edge():
    """Test edge cases."""
    # Single-class degenerate distribution
    probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    result = cumulative_probability(probs, m=1)
    assert isinstance(result, dict)
    assert "value" in result
    # P(y <= 1) = first element
    assert abs(result["value"] - 1.0) < 1e-12

    # All probability mass on the first m classes
    probs2 = np.array([0.5, 0.3, 0.1, 0.1, 0.0, 0.0])
    m = 3
    result2 = cumulative_probability(probs2, m=m)
    expected2 = 0.5 + 0.3 + 0.1
    assert abs(result2["value"] - expected2) < 1e-12
