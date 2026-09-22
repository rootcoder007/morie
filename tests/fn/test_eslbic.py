"""Tests for eslbic.esl_bic_score."""

from morie.fn import _array_core as np

from morie.fn.eslbic import esl_bic_score


def test_eslbic_basic():
    """Test basic functionality."""
    loglik = float(np.random.default_rng(42).normal(0, 1))
    d = 5
    N = 100
    result = esl_bic_score(loglik, d, N)
    assert isinstance(result, dict)
    assert "estimate" in result
    expected = -2.0 * loglik + d * np.log(N)
    assert result["estimate"] == expected
    assert result["penalty"] == d * np.log(N)
    assert result["aic_penalty"] == 2.0 * d
    assert result["loglik"] == loglik
    assert result["d"] == d
    assert result["N"] == N
    assert result["method"] == "BIC = -2 log L + d log N"
    assert result["penalises_more_than_aic"] == (d * np.log(N) > 2.0 * d)


def test_eslbic_edge():
    """Test edge cases."""
    loglik = float(np.random.default_rng(42).normal(0, 1))
    d = 5
    N = 100
    result = esl_bic_score(loglik, d, N)
    assert isinstance(result, dict)
    assert "estimate" in result
