"""Tests for rgcoh.rangayyan_coherence.

Spec: Rangayyan & Krishnan (2024) Sec 4.5.1 "Coherence analysis of EEG
channels", p.235. Magnitude-squared coherence is bounded in [0, 1] and is
identically 1 when one signal is a linear time-invariant transform of the
other -- both properties are pinned here.
"""

import numpy as np
import pytest

from morie.fn.rgcoh import rangayyan_coherence


def test_rgcoh_self_coherence_is_unity():
    x = np.random.default_rng(3).standard_normal(1024)
    c = np.asarray(rangayyan_coherence(x, x, fs=100.0)["coherence"], dtype=float)
    assert np.allclose(c, 1.0, atol=1e-8)


def test_rgcoh_is_bounded_in_unit_interval():
    rng = np.random.default_rng(4)
    r = rangayyan_coherence(rng.standard_normal(1024), rng.standard_normal(1024), fs=100.0)
    c = np.asarray(r["coherence"], dtype=float)
    assert np.all(c >= -1e-12)
    assert np.all(c <= 1.0 + 1e-12)


def test_rgcoh_scaled_copy_stays_fully_coherent():
    # y = a*x is an LTI transform, so MSC is 1 at every frequency.
    x = np.random.default_rng(5).standard_normal(1024)
    c = np.asarray(rangayyan_coherence(x, 2.5 * x, fs=100.0)["coherence"], dtype=float)
    assert np.allclose(c, 1.0, atol=1e-8)


def test_rgcoh_independent_signals_are_less_coherent_than_identical_ones():
    rng = np.random.default_rng(6)
    x, y = rng.standard_normal(4096), rng.standard_normal(4096)
    indep = rangayyan_coherence(x, y, fs=100.0)["mean_coherence"]
    same = rangayyan_coherence(x, x, fs=100.0)["mean_coherence"]
    assert indep < same
    assert same == pytest.approx(1.0, abs=1e-8)


def test_rgcoh_peak_frequency_is_within_the_analysed_band():
    rng = np.random.default_rng(7)
    r = rangayyan_coherence(rng.standard_normal(1024), rng.standard_normal(1024), fs=100.0)
    freqs = np.asarray(r["freqs"], dtype=float)
    assert freqs[0] <= r["peak_freq"] <= freqs[-1]
    assert 0.0 <= r["peak_coherence"] <= 1.0 + 1e-12
