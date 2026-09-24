"""Tests for gb_odi.gibbons_odds_ratio."""

from morie.fn import _array_core as np

from morie.fn.gb_odi import gibbons_odds_ratio


def test_gb_odi_basic():
    """Test basic functionality with a 2x2 contingency table."""
    rng = np.random.default_rng(42)
    # 2x2 contingency table with positive integer counts
    table = [
        [int(rng.integers(10, 50)), int(rng.integers(10, 50))],
        [int(rng.integers(10, 50)), int(rng.integers(10, 50))]
    ]
    result = gibbons_odds_ratio(table)
    assert isinstance(result, dict)
    # The result should contain test statistic and/or p-value
    assert "statistic" in result or "p_value" in result


def test_gb_odi_edge():
    """Test edge cases with small counts (Gibbons method handles small samples)."""
    rng = np.random.default_rng(42)
    # Small 2x2 table - Gibbons method is designed for this
    table = [[1, 2], [3, 4]]
    result = gibbons_odds_ratio(table)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result
