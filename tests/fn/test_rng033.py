"""Tests for rng033.rangayyan_ch3_causal_convolution_alt."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsasig import rangayyan_ch3_causal_convolution
from morie.fn.bsasig import rangayyan_ch3_causal_convolution_alt


def test_rng033_basic():
    t = np.linspace(0, 3, 301)
    x = np.exp(-t)
    h = np.sin(t)
    dt = t[1] - t[0]
    a = rangayyan_ch3_causal_convolution_alt(x, h, dt=dt)
    b = rangayyan_ch3_causal_convolution(x, h, dt=dt)
    assert a["y"] == pytest.approx(b["y"])  # convolution commutes


def test_rng033_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_causal_convolution_alt([1.0, 2.0], [1.0])  # grid mismatch
