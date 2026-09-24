"""Tests for klmcmc.kl_mcmc_diagnostic."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.klmcmc import kl_mcmc_diagnostic


def _std_normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def test_klmcmc_basic():
    """Test basic functionality."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = kl_mcmc_diagnostic(chain, _std_normal_pdf, bins=20, lo=-5.0, hi=5.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "kl" in result
    assert "p" in result
    assert "q" in result
    assert "unsupported_bins" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 100
    assert len(result["p"]) == 20
    assert len(result["q"]) == 20
    assert abs(sum(result["p"]) - 1.0) < 1e-9
    assert abs(sum(result["q"]) - 1.0) < 1e-9


def test_klmcmc_edge():
    """Test edge cases."""
    # a chain with fewer than two draws is invalid per the docstring
    with pytest.raises(ValueError):
        kl_mcmc_diagnostic([0.5], _std_normal_pdf, bins=20, lo=-5.0, hi=5.0)
