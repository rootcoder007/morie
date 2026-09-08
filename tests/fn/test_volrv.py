"""Tests for volrv."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volrv import vol_realised_variance


def test_volrv_basic():
    assert vol_realised_variance([0.1, -0.2, 0.3])["rv"] == pytest.approx(0.14)


def test_volrv_edge():
    out = vol_realised_variance([0.1, 0.2, 0.3, 0.4], ["a", "a", "b", "b"])
    assert out["rv"] == pytest.approx([0.05, 0.25])
    with pytest.raises(ValueError):
        vol_realised_variance([0.1])
