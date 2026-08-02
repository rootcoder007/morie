"""Tests for spcoef."""

from morie.fn import _array_core as np
import pytest

from morie.fn.spcoef import spearmans_rho_copula

def test_spcoef_basic():
    out = spearmans_rho_copula("gaussian", 0.6)
    assert out["exact"] is True
    assert out["rho_s"] == pytest.approx(6 / np.pi * np.arcsin(0.3))


def test_spcoef_edge():
    assert spearmans_rho_copula("independence")["rho_s"] == 0.0
    with pytest.raises(ValueError):
        spearmans_rho_copula("clayton", 2.0, n=5)
