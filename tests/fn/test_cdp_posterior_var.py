"""Tests for cdp_posterior_var.cdp_posterior_var."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.cdp_posterior_var import cdp_posterior_var


def test_ghs015_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    alpha = rng.uniform(0.1, 5.0, 10)
    counts = rng.integers(0, 20, 10)
    j = 0
    alpha_total = float(np.sum(alpha))
    result = cdp_posterior_var(alpha, counts, j, alpha_total)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0


def test_ghs015_edge():
    """Test edge cases with j at the last index."""
    rng = np.random.default_rng(42)
    alpha = rng.uniform(0.1, 5.0, 10)
    counts = rng.integers(0, 20, 10)
    j = 9
    alpha_total = float(np.sum(alpha))
    result = cdp_posterior_var(alpha, counts, j, alpha_total)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isfinite(result["value"])
    assert result["value"] >= 0
