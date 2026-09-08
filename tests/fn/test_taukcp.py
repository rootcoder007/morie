"""Tests for taukcp."""

from morie.fn import _array_core as np
import pytest

from morie.fn.taukcp import kendalls_tau_copula

def test_taukcp_basic():
    out = kendalls_tau_copula("clayton", 3.0)
    assert out["tau"] == pytest.approx(0.6)
    assert out["theta_roundtrip"] == pytest.approx(3.0, abs=1e-6)


def test_taukcp_edge():
    assert kendalls_tau_copula("independence")["tau"] == 0.0
    with pytest.raises(ValueError):
        kendalls_tau_copula("weibull", 2.0)
