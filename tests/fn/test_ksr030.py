"""Tests for ksr030 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr030 import kosorok_ch2_brownian_bridge_covariance


def test_ksr030_basic():
    out = kosorok_ch2_brownian_bridge_covariance(0.3, 0.7)
    assert out["covariance"] == pytest.approx(0.3 - 0.21)  # F(s^t) - F(s)F(t)


def test_ksr030_edge():
    # the bridge is tied down at both endpoints
    assert kosorok_ch2_brownian_bridge_covariance(0.0, 0.5)["covariance"] == \
        pytest.approx(0.0)
    assert kosorok_ch2_brownian_bridge_covariance(1.0, 0.5)["covariance"] == \
        pytest.approx(0.0)
