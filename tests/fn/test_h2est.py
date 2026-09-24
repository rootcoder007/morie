"""Tests for h2est.heritability_lmm."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.h2est import heritability_lmm


def test_h2est_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    sigma_g2 = [abs(x) + 0.1 for x in rng.normal(0, 1, 100)]
    sigma_e2 = [abs(x) + 0.1 for x in rng.normal(0, 1, 100)]
    result = heritability_lmm(sigma_g2, sigma_e2)
    payload = result.payload
    assert isinstance(payload, dict)
    assert "estimate" in payload
    assert "h2" in payload
    assert "sigma_p2" in payload
    assert "n" in payload
    assert "method" in payload
    assert payload["n"] == 100
    assert math.isfinite(payload["estimate"])
    assert 0.0 <= payload["estimate"] <= 1.0
    assert len(payload["h2"]) == 100
    assert len(payload["sigma_p2"]) == 100
    for h in payload["h2"]:
        assert 0.0 <= h <= 1.0
        assert math.isfinite(h)
    for sp in payload["sigma_p2"]:
        assert sp > 0.0
        assert math.isfinite(sp)


def test_h2est_edge():
    """Test edge cases."""
    with pytest.raises(ValueError):
        heritability_lmm(-1.0, 1.0)
    with pytest.raises(ValueError):
        heritability_lmm(1.0, -1.0)
