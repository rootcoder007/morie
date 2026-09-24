"""Tests for km148.kamath_ch9_ldm_loss."""

import math

from morie.fn import _array_core as np

from morie.fn.km148 import kamath_ch9_ldm_loss


def test_km148_basic():
    """Test basic functionality with eps_net as prediction array."""
    rng = np.random.default_rng(42)
    epsilon = rng.normal(0, 1, (20, 4))
    z_t = rng.normal(0, 1, (20, 4))
    H_X = rng.normal(0, 1, (20, 2))
    eps_pred = rng.normal(0, 1, (20, 4))
    result = kamath_ch9_ldm_loss(epsilon, z_t, H_X, eps_net=eps_pred, t=1)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 0.0
    assert result["per_sample"]
    assert len(result["per_sample"]) == 20
    assert result["n"] == 20
    assert result["method"]


def test_km148_edge():
    """Test edge cases with eps_net as callable (matching the docstring)."""
    epsilon = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    z_t = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    H_X = np.asarray([[0.0], [0.0]])

    def eps_net(z, t, h):
        return np.zeros_like(epsilon)

    result = kamath_ch9_ldm_loss(epsilon, z_t, H_X, eps_net=eps_net, t=0.5)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["estimate"] == 1.0
    assert result["n"] == 2
    assert len(result["per_sample"]) == 2
    assert all(math.isfinite(v) and v >= 0.0 for v in result["per_sample"])
