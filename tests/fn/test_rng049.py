"""Tests for rng049.rangayyan_ch3_laplace_transform_causal_finite."""

import numpy as np
import pytest

from morie.fn.rng049 import rangayyan_ch3_laplace_transform_causal_finite


def test_rng049_basic():
    # unit pulse on [0, T]: H(s) = (1 - e^{-sT}) / s
    T, dt, s = 2.0, 1e-3, 0.5
    h = np.ones(int(T / dt) + 1)
    result = rangayyan_ch3_laplace_transform_causal_finite(h, s, dt=dt)
    assert result["H"].real == pytest.approx((1 - np.exp(-s * T)) / s, abs=1e-6)
    assert result["T"] == pytest.approx(T)


def test_rng049_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_laplace_transform_causal_finite([1.0], 1.0)  # need >= 2 samples
    with pytest.raises(ValueError):
        rangayyan_ch3_laplace_transform_causal_finite([1.0, 1.0], 1.0, dt=-1.0)
