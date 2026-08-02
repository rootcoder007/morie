"""Tests for volopn."""

from morie.fn import _array_core as np
import pytest

from morie.fn.volopn import _bs_price, vol_implied_volatility_bs


def test_volopn_basic():
    S, K, T, r = 100.0, 100.0, 1.0, 0.01
    price = _bs_price(S, K, T, r, 0.25, "call")
    out = vol_implied_volatility_bs(S, K, T, r, price, "call")
    assert out["implied_vol"] == pytest.approx(0.25, abs=1e-8)


def test_volopn_edge():
    with pytest.raises(ValueError):
        vol_implied_volatility_bs(100, 100, 1.0, 0.01, 101.0, "call")  # above bound
    with pytest.raises(ValueError):
        vol_implied_volatility_bs(100, 100, 0.0, 0.01, 5.0, "call")
