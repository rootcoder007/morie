"""Tests for rgemgfd.rangayyan_emg_fractal_dim."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.rgemgfd import rangayyan_emg_fractal_dim


def _make_force(n_levels=4, fs=100, rest=0.0):
    """Build a piecewise-constant force signal with n_levels above rest."""
    seglen = int(round(fs))  # 1 second
    levels = [float(i + 1) for i in range(n_levels)]
    f = []
    for lvl in levels:
        f.extend([lvl] * seglen)
    f.extend([rest] * seglen)  # trailing rest
    return f


def test_rgemgfd_basic():
    """Test basic functionality."""
    fs = 100.0
    force = _make_force(n_levels=4, fs=fs)
    n = len(force)
    emg = np.random.default_rng(42).normal(0, 1, n)
    result = rangayyan_emg_fractal_dim(emg, force, fs)

    assert isinstance(result, dict)
    # Keys from the return payload
    assert "estimate" in result
    assert "levels" in result
    assert "fd" in result
    assert "slope" in result
    assert "intercept" in result
    assert "r2" in result

    # Shape and type checks
    assert isinstance(result["levels"], list)
    assert isinstance(result["fd"], list)
    assert len(result["levels"]) >= 2
    assert len(result["levels"]) == len(result["fd"])

    # All numeric outputs should be finite
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["slope"])
    assert math.isfinite(result["intercept"])
    assert math.isfinite(result["r2"])


def test_rgemgfd_edge():
    """Test edge case: fewer than two usable levels raises ValueError."""
    fs = 100.0
    seglen = int(round(fs))
    # Only one level above rest_level, plus rest
    force = [1.0] * seglen + [0.0] * seglen
    emg = np.random.default_rng(42).normal(0, 1, len(force))

    with pytest.raises(ValueError):
        rangayyan_emg_fractal_dim(emg, force, fs)
