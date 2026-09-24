"""Tests for rgpsdacf.rangayyan_psd_to_acf."""

import math

from morie.fn import _array_core as np

from morie.fn.bsacorr import rangayyan_psd_to_acf


def test_rgpsdacf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 64
    psd = np.abs(rng.normal(0, 1, n))
    freqs = np.linspace(0.0, 0.5, n)
    result = rangayyan_psd_to_acf(psd, freqs)
    assert isinstance(result, dict)
    assert "acf" in result
    assert "lags" in result
    assert "r0" in result
    assert "method" in result
    assert len(result["acf"]) == len(result["lags"])
    assert math.isfinite(result["r0"])
    assert result["r0"] >= 0


def test_rgpsdacf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 2
    psd = np.abs(rng.normal(0, 1, n))
    result = rangayyan_psd_to_acf(psd)
    assert isinstance(result, dict)
    assert "acf" in result
    assert "lags" in result
    assert "r0" in result
    assert "method" in result
    assert len(result["acf"]) == len(result["lags"])
    assert math.isfinite(result["r0"])
