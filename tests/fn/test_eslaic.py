"""Tests for eslaic.esl_aic_score."""

from morie.fn import _array_core as np

from morie.fn.eslaic import esl_aic_score


def test_eslaic_basic():
    """Test basic functionality."""
    loglik = -100.0
    d = 3
    result = esl_aic_score(loglik, d)
    assert isinstance(result, dict)
    assert "estimate" in result
    expected = -2.0 * loglik + 2.0 * d
    assert result["estimate"] == expected
    assert result["loglik"] == loglik
    assert result["d"] == d
    assert result["method"] == "AIC = -2 log L + 2 d"


def test_eslaic_edge():
    """Test edge cases."""
    loglik = -50.0
    d = 0
    result = esl_aic_score(loglik, d)
    assert isinstance(result, dict)
    expected = -2.0 * loglik + 2.0 * d
    assert result["estimate"] == expected
    assert result["d"] == 0
