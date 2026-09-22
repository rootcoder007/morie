"""Tests for ca9e6.ca_chapter_9_equation_6."""

from morie.fn import _array_core as np

from morie.fn.ca9e6 import ca_chapter_9_equation_6


def test_ca9e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Three groups with at least 2 observations each, as required by the formula.
    groups = [
        rng.normal(0, 1, 5),
        rng.normal(1, 1, 6),
        rng.normal(2, 1, 4),
    ]
    result = ca_chapter_9_equation_6(groups)
    assert isinstance(result, dict)
    assert "ms_within" in result
    assert result["method"] == "Weisburd et al. (2022) eq. (9.6)"
    assert "value" in result and result["value"] == result["ms_within"]

    # Independent computation of the formula:
    # MS_within = sum_j sum_i (y_ij - ybar_j)^2 / (N - a)
    a = len(groups)
    N = sum(len(g) for g in groups)
    expected = sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / (N - a)
    assert result["ms_within"] == expected


def test_ca9e6_edge():
    """Test edge cases: minimum valid input (2 groups, 2 obs each)."""
    rng = np.random.default_rng(42)
    groups = [
        rng.normal(0, 1, 2),
        rng.normal(1, 1, 2),
    ]
    result = ca_chapter_9_equation_6(groups)
    assert isinstance(result, dict)
    assert "ms_within" in result

    a = len(groups)
    N = sum(len(g) for g in groups)
    expected = sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / (N - a)
    assert result["ms_within"] == expected
