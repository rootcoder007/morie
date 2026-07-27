"""Tests for rng053.rangayyan_ch3_z_transform_fir."""

import pytest

from morie.fn.rng053 import rangayyan_ch3_z_transform_fir


def test_rng053_basic():
    h = [1.0, -0.5]
    assert rangayyan_ch3_z_transform_fir(h, 1.0)["H"].real == pytest.approx(0.5)
    assert rangayyan_ch3_z_transform_fir(h, -1.0)["H"].real == pytest.approx(1.5)
    assert rangayyan_ch3_z_transform_fir(h, 2.0)["N"] == 2


def test_rng053_edge():
    with pytest.raises(ValueError):
        rangayyan_ch3_z_transform_fir([1.0], 0.0)  # z = 0 is a pole
    with pytest.raises(ValueError):
        rangayyan_ch3_z_transform_fir([], 1.0)
