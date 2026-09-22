"""Tests for bayfac.bayes_factor."""

from morie.fn import _array_core as np

from morie.fn.bayfac import bayes_factor


def test_bayfac_basic():
    """Test basic functionality."""
    log_evidence_1 = float(np.random.default_rng(42).normal(0, 1, 1)[0])
    log_evidence_2 = float(np.random.default_rng(42).normal(0, 1, 1)[0])
    expected_log_bf = log_evidence_1 - log_evidence_2
    expected_two_log_bf = 2.0 * expected_log_bf
    expected_log10_bf = expected_log_bf / np.log(10.0)
    expected_bf = np.exp(expected_log_bf)
    result = bayes_factor(log_evidence_1, log_evidence_2)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    assert "bf" in result
    assert "log_bf" in result
    assert "two_log_bf" in result
    assert "log10_bf" in result
    assert "category" in result
    assert "favours" in result
    assert "bf_21" in result
    assert result["log_bf"] == expected_log_bf
    assert result["two_log_bf"] == expected_two_log_bf
    assert result["log10_bf"] == expected_log10_bf
    if expected_log_bf > 709.0:
        assert result["bf"] == float("inf")
    elif expected_log_bf < -745.0:
        assert result["bf"] == 0.0
    else:
        assert result["bf"] == expected_bf


def test_bayfac_edge():
    """Test edge cases."""
    log_evidence_1 = float(np.random.default_rng(42).normal(0, 1, 1)[0])
    log_evidence_2 = float(np.random.default_rng(42).normal(0, 1, 1)[0])
    result = bayes_factor(log_evidence_1, log_evidence_2)
    assert isinstance(result, dict)
