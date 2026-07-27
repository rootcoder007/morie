"""Tests for volpow."""

import numpy as np
import pytest

from morie.fn.volpow import vol_power_variation


def test_volpow_basic():
    r = np.array([0.1, -0.2, 0.3])
    out = vol_power_variation(r, p=1.0)
    assert out["pv"] == pytest.approx(0.6)
    p2 = vol_power_variation(r, p=2.0)
    assert p2["pv_standardised"] == pytest.approx((r**2).sum())  # mu_2 = 1


def test_volpow_edge():
    with pytest.raises(ValueError):
        vol_power_variation([0.1, 0.2], p=-1.0)
