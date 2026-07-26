"""Tests for rgenv.rangayyan_envelope.

Spec: Rangayyan & Krishnan (2024) Sec 5.5.3 "The envelogram", p.281, in
Sec 5.5 "Envelope Extraction and Analysis", p.277. The envelope is the
magnitude of the analytic signal, so it bounds |x| from above and recovers
the modulator of an AM signal.
"""

import numpy as np
import pytest

from morie.fn.rgenv import rangayyan_envelope


def test_rgenv_envelope_bounds_the_signal():
    x = np.random.default_rng(21).standard_normal(512)
    env = np.asarray(rangayyan_envelope(x)["envelope"], dtype=float)
    assert np.all(env >= np.abs(x) - 1e-9)


def test_rgenv_constant_amplitude_sine_has_flat_envelope():
    # |analytic(sin)| = 1 everywhere; edges are excluded because the Hilbert
    # transform of a finite record is not exact at the boundaries.
    t = np.arange(4096) / 512.0
    env = np.asarray(rangayyan_envelope(np.sin(2 * np.pi * 20.0 * t))["envelope"], dtype=float)
    assert np.allclose(env[200:-200], 1.0, atol=0.02)


def test_rgenv_recovers_an_am_modulator():
    # x(t) = m(t) cos(2 pi fc t) with m slow and positive -> envelope ~ m(t)
    fs, n = 1000.0, 4096
    t = np.arange(n) / fs
    m = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    env = np.asarray(rangayyan_envelope(m * np.cos(2 * np.pi * 150.0 * t))["envelope"], dtype=float)
    assert np.allclose(env[300:-300], m[300:-300], atol=0.05)


def test_rgenv_envelope_is_nonnegative():
    env = np.asarray(rangayyan_envelope(np.random.default_rng(22).standard_normal(256))["envelope"], dtype=float)
    assert np.all(env >= 0.0)


def test_rgenv_instantaneous_frequency_tracks_a_pure_tone():
    fs, n, f0 = 1000.0, 4096, 50.0
    t = np.arange(n) / fs
    r = rangayyan_envelope(np.cos(2 * np.pi * f0 * t))
    inst = np.asarray(r["instantaneous_freq"], dtype=float)
    # instantaneous_freq is in cycles/sample; scale by fs for Hz
    assert float(np.median(inst[300:-300])) * fs == pytest.approx(f0, rel=0.05)
