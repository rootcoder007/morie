"""Tests for rng032.rangayyan_ch3_causal_convolution."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rng032 import rangayyan_ch3_causal_convolution


def test_rng032_basic():
    # unit step convolved with a unit step gives y(t) = t
    t = np.linspace(0, 4, 401)
    one = np.ones_like(t)
    result = rangayyan_ch3_causal_convolution(one, one, dt=t[1] - t[0])
    assert result["y"] == pytest.approx(t, abs=1e-9)
    assert result["y"][0] == 0.0


def test_rng032_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_causal_convolution([1.0], [1.0])  # need >= 2 samples
    with pytest.raises(ValueError):
        rangayyan_ch3_causal_convolution([1.0, 2.0], [1.0, 2.0], dt=0.0)
