"""Tests for ca6e6.ca_chapter_6_equation_6."""

from morie.fn import _array_core as np

from morie.fn.ca6e6 import ca_chapter_6_equation_6


def test_ca6e6_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # -2LL_null must be greater than -2LL_full for a meaningful model fit
    neg2ll_full = float(np.sum(rng.normal(500.0, 1.0, 100)))
    neg2ll_null = neg2ll_full + float(np.sum(rng.normal(10.0, 1.0, 100)))
    result = ca_chapter_6_equation_6(neg2ll_null, neg2ll_full)
    assert isinstance(result, dict)
    # Headline key per docstring is 'value'
    assert "value" in result
    # Formula: Model chi2 = (-2LL_null) - (-2LL_full)
    expected_value = neg2ll_null - neg2ll_full
    assert result["value"] == expected_value
    # The full payload should include the method label
    assert result["method"] == "Weisburd et al. (2022) eq. (6.6)"


def test_ca6e6_edge():
    """Test edge cases."""
    # When the two log-likelihoods are equal, the model chi-square should be zero
    neg2ll_null = 250.0
    neg2ll_full = 250.0
    result = ca_chapter_6_equation_6(neg2ll_null, neg2ll_full)
    assert isinstance(result, dict)
    assert "value" in result
    # Independent computation of the documented formula
    assert result["value"] == neg2ll_null - neg2ll_full
    assert result["value"] == 0.0
