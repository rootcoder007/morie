"""Tests for rgcepsp.rangayyan_cepstrum_pitch."""

import math

from morie.fn import _array_core as np

from morie.fn.bsacep import rangayyan_cepstrum_pitch


def test_rgcepsp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    fs = 100.0
    f0_range = (50.0, 500.0)
    result = rangayyan_cepstrum_pitch(x, fs, f0_range)
    assert isinstance(result, dict)
    for key in ("f0", "period_s", "quefrency", "cepstrum", "peak_value", "method"):
        assert key in result
    f0 = result["f0"]
    assert math.isfinite(f0)
    assert f0_range[0] <= f0 <= f0_range[1]
    assert math.isclose(1.0 / result["period_s"], f0)


def test_rgcepsp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 16)
    fs = 100.0
    f0_range = (50.0, 500.0)
    result = rangayyan_cepstrum_pitch(x, fs, f0_range)
    assert isinstance(result, dict)
    assert "f0" in result
    assert "cepstrum" in result
