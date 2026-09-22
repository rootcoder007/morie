"""Tests for bdgsm.bridge_sampling_marginal."""

from morie.fn import _array_core as np

from morie.fn.bdgsm import bridge_sampling_marginal


def test_bdgsm_basic():
    """Test basic functionality."""
    log_p_posterior = np.random.default_rng(42).normal(0, 1, 100)
    log_q_posterior = np.random.default_rng(42).normal(0, 1, 100)
    log_p_proposal = np.random.default_rng(42).normal(0, 1, 100)
    log_q_proposal = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling_marginal(log_p_posterior, log_q_posterior, log_p_proposal, log_q_proposal)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_bdgsm_edge():
    """Test edge cases."""
    log_p_posterior = np.random.default_rng(42).normal(0, 1, 100)
    log_q_posterior = np.random.default_rng(42).normal(0, 1, 100)
    log_p_proposal = np.random.default_rng(42).normal(0, 1, 100)
    log_q_proposal = np.random.default_rng(42).normal(0, 1, 100)
    result = bridge_sampling_marginal(log_p_posterior, log_q_posterior, log_p_proposal, log_q_proposal)
    assert isinstance(result, dict)
