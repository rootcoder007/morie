"""Tests for b205.burkov_lm_ch2_perplexity."""

from morie.fn import _array_core as np

from morie.fn.b205 import burkov_lm_ch2_perplexity


def test_b205_basic():
    """Test basic functionality."""
    log_probs = np.log(np.array([0.5, 0.5]))
    k = 1
    t = np.array([0, 1])
    result = burkov_lm_ch2_perplexity(log_probs, k, t)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
    assert "perplexity" in result
    assert "cross_entropy" in result
    assert "bits_per_token" in result
    assert "n_tokens" in result
    assert "uniform_ceiling" in result
    expected_perplexity = float(np.exp(-np.mean(log_probs)))
    assert abs(result["perplexity"] - expected_perplexity) < 1e-12
    assert abs(result["estimate"] - expected_perplexity) < 1e-12
    expected_ce = float(-np.mean(log_probs))
    assert abs(result["cross_entropy"] - expected_ce) < 1e-12
    expected_bpt = float(expected_ce / np.log(2.0))
    assert abs(result["bits_per_token"] - expected_bpt) < 1e-12
    assert result["n_tokens"] == 2
    assert result["context_window"] == 1


def test_b205_edge():
    """Test edge cases."""
    log_probs = np.log(np.array([0.5, 0.5]))
    k = 1
    t = np.array([0, 1])
    result = burkov_lm_ch2_perplexity(log_probs, k, t)
    assert isinstance(result, dict)
    assert "perplexity" in result
    assert result["n_tokens"] == 2
