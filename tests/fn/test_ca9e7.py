"""Tests for ca9e7.ca_chapter_9_equation_7."""

from morie.fn import _array_core as np

from morie.fn.ca9e7 import ca_chapter_9_equation_7


def test_ca9e7_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # ca_chapter_9_equation_7 implements one-way ANOVA: it needs at least
    # two groups, each with at least two observations.
    groups = [rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)]
    result = ca_chapter_9_equation_7(groups)

    # The implementation returns a RichResult (dict subclass). The headline
    # statistic is exposed under the key 'f' (and mirrored as 'value').
    assert isinstance(result, dict)
    assert "f" in result
    assert "value" in result
    assert result["f"] == result["value"]

    # Independent computation of one-way ANOVA F from the formula
    # F = MS_between / MS_within.
    all_means = [np.mean(g) for g in groups]
    grand_mean = np.mean([np.mean(g) for g in groups])
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    df_between = k - 1
    df_within = n_total - k

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    expected_f = ms_between / ms_within

    assert np.isclose(result["f"], expected_f)


def test_ca9e7_edge():
    """Test edge cases: minimally specified input (two groups of two)."""
    groups = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
    ]
    result = ca_chapter_9_equation_7(groups)
    assert isinstance(result, dict)
    assert "f" in result
    assert result["f"] >= 0.0
