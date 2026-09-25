"""Tests for spspec.schabenberger_spectral_representation (eqs 2.26-2.27)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.spspec import schabenberger_spectral_representation


def test_spspec_basic():
    """C(h) = sum_j sigma2_j cos(omega_j h), C(0) = sum sigma2_j, and the
    realisation is mu + sum_j sqrt(2 sigma2_j) cos(omega_j s + phi_j) with
    the phases the seeded uniform(0, 2 pi) draws."""
    om = [0.5, 1.3, 2.0]
    s2 = [1.0, 0.5, 0.25]
    h = [0.0, 0.7, 2.5]
    r = schabenberger_spectral_representation(h, sigma2=s2, omega=om, mu=3.0, seed=11)
    for hh, c in zip(h, r["covariance"]):
        assert float(c) == pytest.approx(sum(a * math.cos(w * hh) for a, w in zip(s2, om)), abs=1e-15)
    assert r["variance"] == r["sum_sigma2"] == 1.75
    ph = [float(v) for v in np.random.default_rng(11).uniform(0.0, 2 * math.pi, 3)]
    for hh, z in zip(h, r["realisation"]):
        assert float(z) == pytest.approx(3.0 + sum(math.sqrt(2 * a) * math.cos(w * hh + p)
                                                   for a, w, p in zip(s2, om, ph)), abs=1e-12)


def test_spspec_edge():
    """Averaging Z(0) Z(h) over a uniform grid of phase shifts phi + t
    (t = 2 pi k / m, m > 2) recovers C(h) exactly, the random-phase
    identity behind eq. (2.27); bad inputs raise."""
    om, s2 = [0.9], [2.0]
    m = 8
    for hh in (0.0, 1.1):
        avg = sum(2 * s2[0] * math.cos(2 * math.pi * k / m) * math.cos(om[0] * hh + 2 * math.pi * k / m)
                  for k in range(m)) / m
        c = schabenberger_spectral_representation([hh], sigma2=s2, omega=om)["covariance"]
        assert float(c[0]) == pytest.approx(avg, abs=1e-14)
    with pytest.raises(ValueError):
        schabenberger_spectral_representation([0.0], sigma2=[1.0, 2.0], omega=[1.0])
    with pytest.raises(ValueError):
        schabenberger_spectral_representation([0.0], sigma2=[-1.0], omega=[1.0])


