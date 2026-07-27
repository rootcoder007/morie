"""Tests for volsd."""

import numpy as np
import pytest

from morie.fn.volsd import vol_simple_diff


def test_volsd_basic():
    out = vol_simple_diff(np.array([1.0, 1.0, 3.0]), window=2)
    assert np.isnan(out["sigma2"][0])
    assert out["sigma2"][2] == pytest.approx(5.0)


def test_volsd_edge():
    with pytest.raises(ValueError):
        vol_simple_diff(np.ones(3), window=1)
    with pytest.raises(ValueError):
        vol_simple_diff(np.ones(3), window=5)
