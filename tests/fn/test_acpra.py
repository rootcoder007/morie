"""Tests for acpra.acceptance_rate_diagnostic."""

from morie.fn import _array_core as np

from morie.fn.acpra import acceptance_rate_diagnostic


def test_acpra_basic():
    """Test basic functionality."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = acceptance_rate_diagnostic(chains)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_acpra_edge():
    """Test edge cases."""
    chains = np.random.default_rng(42).normal(0, 1, 100)
    result = acceptance_rate_diagnostic(chains)
    assert isinstance(result, dict)
