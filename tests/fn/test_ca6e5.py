"""Tests for ca6e5.ca_chapter_6_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca6e5 import ca_chapter_6_equation_5


def test_ca6e5_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # neg2ll_null and neg2ll_full must be scalars (per the formula:
    # Model chi2 = (-2LL_null) - (-2LL_full)). Use a value for the null
    # model larger than the full model so that the chi-square is positive.
    neg2ll_null = float(150.0)
    neg2ll_full = float(120.0)
    result = ca_chapter_6_equation_5(neg2ll_null, neg2ll_full)
    assert isinstance(result, dict)
    # The documented headline key is 'value'.
    assert "value" in result
    # Numeric expectation computed independently from the documented formula.
    expected = neg2ll_null - neg2ll_full
    assert result["value"] == expected


def test_ca6e5_edge():
    """Test edge cases."""
    # Zero chi-square when null and full models have equal -2 log-likelihoods.
    neg2ll_null = float(75.5)
    neg2ll_full = float(75.5)
    result = ca_chapter_6_equation_5(neg2ll_null, neg2ll_full)
    assert isinstance(result, dict)
    assert "value" in result
    assert result["value"] == 0.0
