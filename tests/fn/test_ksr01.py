"""Tests for ksr01.kosorok_empirical_process."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr01 import kosorok_empirical_process


def test_ksr01_identity_function_gives_root_n_mean_deviation():
    """G_n(f) = sqrt(n)(P_n - P)f with f = identity and P f = mu0 is
    exactly sqrt(n)(xbar - mu0) -- computable to the last digit."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    r = kosorok_empirical_process(x, f=lambda v: v, mu0=2.0)
    assert float(r["estimate"]) == pytest.approx(np.sqrt(4) * (2.5 - 2.0), rel=1e-12)


def test_ksr01_centered_data_gives_zero():
    x = np.array([-1.0, 1.0, -2.0, 2.0])
    r = kosorok_empirical_process(x, f=lambda v: v, mu0=0.0)
    assert float(r["estimate"]) == pytest.approx(0.0, abs=1e-12)


def test_ksr01_scales_as_root_n():
    """Duplicating the sample doubles n and multiplies G_n by sqrt(2)
    (the empirical mean is unchanged)."""
    x = np.array([1.0, 3.0, 5.0])
    a = float(kosorok_empirical_process(x, f=lambda v: v, mu0=0.0)["estimate"])
    b = float(kosorok_empirical_process(np.tile(x, 2), f=lambda v: v, mu0=0.0)["estimate"])
    assert b == pytest.approx(np.sqrt(2) * a, rel=1e-12)
