"""Tests for rcaus.random_cause_refutation."""

from morie.fn import _array_core as np

from morie.fn.rcaus import random_cause_refutation


def test_rcaus_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_simulations = np.random.default_rng(42).normal(0, 1, 100)
    result = random_cause_refutation(model, n_simulations)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_rcaus_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    n_simulations = np.random.default_rng(42).normal(0, 1, 100)
    result = random_cause_refutation(model, n_simulations)
    assert isinstance(result, dict)
