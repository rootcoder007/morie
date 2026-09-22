"""Tests for gb_fxe.gibbons_fisher_one_sided."""

from morie.fn import _array_core as np

from morie.fn.gb_fxe import gibbons_fisher_one_sided


def test_gb_fxe_basic():
    """Test basic functionality."""
    table = [[3, 1], [1, 3]]
    result = gibbons_fisher_one_sided(table)
    # RichResult behaves like a dict for the test's purposes.
    assert isinstance(result, dict)
    # Documented keys.
    for key in ("p_value", "p_greater", "p_less", "prob",
                "statistic", "mean", "method"):
        assert key in result

    a, b = 3, 1
    c, d = 1, 3
    r1 = a + b
    r2 = c + d
    c1 = a + c
    nn = r1 + r2
    # Reproduce the formula by plain arithmetic, independent of the function.
    from math import comb, isclose
    den = comb(nn, c1)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    probs = {k: comb(r1, k) * comb(r2, c1 - k) / den for k in range(lo, hi + 1)}
    pg_expected = sum(p for k, p in probs.items() if k >= a)
    pl_expected = sum(p for k, p in probs.items() if k <= a)
    prob_expected = probs[a]
    mean_expected = r1 * c1 / nn

    assert isclose(result["p_greater"], pg_expected)
    assert isclose(result["p_less"], pl_expected)
    assert isclose(result["prob"], prob_expected)
    assert result["statistic"] == a
    assert isclose(result["mean"], mean_expected)
    # Default alternative is "greater".
    assert isclose(result["p_value"], min(1.0, pg_expected))


def test_gb_fxe_edge():
    """Test edge cases (alternative='less')."""
    table = [[3, 1], [1, 3]]
    result = gibbons_fisher_one_sided(table, alternative="less")
    assert isinstance(result, dict)
    assert "p_value" in result

    a, b = 3, 1
    c, d = 1, 3
    r1 = a + b
    r2 = c + d
    c1 = a + c
    nn = r1 + r2
    from math import comb, isclose
    den = comb(nn, c1)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    probs = {k: comb(r1, k) * comb(r2, c1 - k) / den for k in range(lo, hi + 1)}
    pl_expected = sum(p for k, p in probs.items() if k <= a)
    assert isclose(result["p_value"], min(1.0, pl_expected))
