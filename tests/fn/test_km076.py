"""Tests for km076.kamath_ch5_dpo_loss."""

from morie.fn import _array_core as np
import math
import pytest

from morie.fn.km076 import kamath_ch5_dpo_loss


def test_km076_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    pi_theta = [[rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)] for _ in range(n)]
    pi_ref = [[rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)] for _ in range(n)]
    beta = 0.8
    result = kamath_ch5_dpo_loss(pi_theta, pi_ref, beta)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
    assert len(result["margins"]) == n
    assert math.isfinite(result["estimate"])


def test_km076_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        kamath_ch5_dpo_loss([], [], 1.0)
