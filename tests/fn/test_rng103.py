"""Tests for rng103.rangayyan_ch3_integral_causal."""

import numpy as np
import pytest

from morie.fn.rng103 import rangayyan_ch3_integral_causal


def test_rng103_basic():
    t = np.linspace(0, 2, 401)
    result = rangayyan_ch3_integral_causal(t, dt=t[1] - t[0])
    assert result["y"] == pytest.approx(t**2 / 2, abs=1e-9)
    assert result["total"] == pytest.approx(2.0, abs=1e-6)


def test_rng103_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_integral_causal([1.0])
    with pytest.raises(ValueError):
        rangayyan_ch3_integral_causal([1.0, 2.0], dt=0.0)
