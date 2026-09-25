"""Tests for hrzdeconv.horowitz_deconvolution_density (Horowitz sec. 5.1)."""

import math

import pytest

from morie.fn.hrzdeconv import horowitz_deconvolution_density


W = [math.sin(1.7 * k) * 1.3 + 0.2 * math.cos(0.3 * k) for k in range(60)]


def _fhat(u, s, h):
    """(1 / 2 pi) int_{-1/h}^{1/h} Re[psi_W(tau) e^{-i tau u}] (1 - (h tau)^2)^3
    / psi_eps(tau) d tau with the normal error cf, by the trapezoid rule on
    the same 2001-point grid."""
    T = 1.0 / h
    taus = [-T + 2 * T * k / 2000 for k in range(2001)]
    vals = [sum(math.cos(t * (w - u)) for w in W) / len(W) * math.exp(0.5 * s * s * t * t)
            * (1 - (t / T) ** 2) ** 3 for t in taus]
    step = 2 * T / 2000
    return step * (sum(vals) - 0.5 * (vals[0] + vals[-1])) / (2 * math.pi)


def test_hrzdeconv_basic():
    """Normal error: the default cut-off is h = sigma / sqrt(log n) and
    the density is the damped Fourier inversion recomputed here."""
    r = horowitz_deconvolution_density(W, 0.5, grid=[-1.0, 0.0, 0.7])
    h = 0.5 / math.sqrt(math.log(60))
    assert r["bandwidth"] == pytest.approx(h, rel=1e-15)
    assert r["regime"] == "supersmooth"
    assert [float(v) for v in r["density"]] == pytest.approx([_fhat(u, 0.5, h) for u in (-1.0, 0.0, 0.7)], rel=1e-9, abs=1e-12)
    lap = horowitz_deconvolution_density(W, 0.5, grid=[0.0], error="laplace")
    assert lap["bandwidth"] == pytest.approx(60 ** -0.2, rel=1e-15)
    assert lap["regime"] == "ordinary smooth"


def test_hrzdeconv_edge():
    """Unknown error laws, sigma <= 0 and fewer than 8 points raise."""
    with pytest.raises(ValueError):
        horowitz_deconvolution_density(W, 0.5, error="cauchy")
    with pytest.raises(ValueError):
        horowitz_deconvolution_density(W, 0.0)
    with pytest.raises(ValueError):
        horowitz_deconvolution_density(W[:5], 0.5)
