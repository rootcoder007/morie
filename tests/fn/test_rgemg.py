"""Tests for rgemg.rangayyan_emg_rms.

Spec: Rangayyan & Krishnan (2024) Sec 5.6.1, pp.283-284. Eq (5.23) is the
global RMS; eq (5.24) is the running RMS this function computes,

    RMS(n) = [ (1/M) sum_{k=0}^{M-1} x^2(n-k) ]^(1/2),

which is CAUSAL and therefore undefined for n < M-1.
"""

import numpy as np
import pytest

from morie.fn.rgemg import rangayyan_emg_rms


def test_rgemg_constant_signal_gives_its_own_magnitude():
    # RMS of a constant a over any window is |a| exactly (eq 5.24).
    r = rangayyan_emg_rms(np.full(100, 3.0), window=10)
    rms = np.asarray(r["rms"], dtype=float)
    assert np.allclose(rms[9:], 3.0)


def test_rgemg_matches_hand_computed_window():
    # window of 2 over [3, 4]: sqrt((9 + 16)/2) = sqrt(12.5)
    x = np.array([0.0, 3.0, 4.0])
    rms = np.asarray(rangayyan_emg_rms(x, window=2)["rms"], dtype=float)
    assert rms[2] == pytest.approx(np.sqrt(12.5))
    assert rms[1] == pytest.approx(np.sqrt(4.5))  # (0 + 9)/2


def test_rgemg_warmup_is_undefined_not_backfilled():
    # Eq (5.24) is causal: the first M-1 samples have no defined value.
    x = np.arange(1.0, 21.0)
    rms = np.asarray(rangayyan_emg_rms(x, window=5)["rms"], dtype=float)
    assert np.all(np.isnan(rms[:4]))
    assert np.isfinite(rms[4])


def test_rgemg_envelope_does_not_rise_before_onset():
    # The regression this guards: back-filling the warm-up with rms[W-1]
    # used a value computed from samples in the FUTURE of those positions,
    # so a signal silent until sample 20 reported 0.7651 at sample 0.
    x = np.zeros(400)
    x[20:] = np.random.default_rng(1).standard_normal(380)
    rms = np.asarray(rangayyan_emg_rms(x, window=64)["rms"], dtype=float)
    active = np.nonzero(np.nan_to_num(rms) > 1e-12)[0]
    assert active.size > 0
    assert active[0] >= 20


def test_rgemg_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="window"):
        rangayyan_emg_rms(np.arange(10.0), window=0)
