"""Tests for rgztf.rangayyan_z_transform."""

import pytest

from morie.fn.rgztf import rangayyan_z_transform


def test_rgztf_basic():
    out = rangayyan_z_transform([1.0, 0.5, 0.25], z=2.0)
    assert out["degree"] == 2
    assert out["H"].real == pytest.approx(1 + 0.5 / 2 + 0.25 / 4)


def test_rgztf_edge():
    assert rangayyan_z_transform([1.0])["H"] is None
    with pytest.raises(ValueError):
        rangayyan_z_transform([])
