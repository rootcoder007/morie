"""Tests for ksr049.kosorok_ch2_z_master_linearization."""

import pytest

from morie.fn.ksr049 import kosorok_ch2_z_master_linearization


def test_ksr049_basic():
    """The mean as a Z-estimator: Psi_n(theta) = xbar - theta, Psi(theta) =
    mu - theta, Psi_dot = -1. Eq. (2.13)'s residual is then exactly zero:
    -sqrt(n)(xbar - mu) + sqrt(n)(xbar - mu)."""
    x = [0.3, 1.9, -0.4, 1.1, 0.8, 2.2]
    xbar, mu = sum(x) / 6, 0.5
    r = kosorok_ch2_z_master_linearization(
        [[-1.0]], lambda th, t: xbar - th[0], lambda th, t: mu - th[0], [xbar], [mu], 6)
    assert r["residual_norm"] == pytest.approx(0.0, abs=1e-14)
    assert r["derivative_invertible"] is True


def test_ksr049_edge():
    """A singular derivative is reported, and theta shapes must agree."""
    r = kosorok_ch2_z_master_linearization(
        [[0.0]], lambda th, t: 0.0, lambda th, t: 0.0, [1.0], [1.0], 4)
    assert r["derivative_invertible"] is False
    with pytest.raises(ValueError, match="same length"):
        kosorok_ch2_z_master_linearization([[1.0]], lambda th, t: 0.0, lambda th, t: 0.0, [1.0, 2.0], [1.0], 4)


