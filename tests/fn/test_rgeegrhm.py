"""Tests for rgeegrhm.rangayyan_eeg_rhythm_detect."""

import math

import pytest

from morie.fn.bsacorr import rangayyan_eeg_rhythm_detect


def test_rgeegrhm_basic():
    """A 10 Hz rhythm sampled at 100 Hz has its first ACF peak at lag 10,
    inside the 8-13 Hz alpha band, so alpha is present at 10 Hz."""
    x = [math.sin(2 * math.pi * 10 * n / 100) for n in range(400)]
    r = rangayyan_eeg_rhythm_detect(x, 100.0)
    assert r["present"] is True
    assert r["peak_lag"] == 10 and r["frequency_hz"] == 10.0
    assert r["lag_range"] == (7, 13)


def test_rgeegrhm_edge():
    """A 25 Hz (beta) rhythm is not alpha, but is found with the beta band."""
    x = [math.sin(2 * math.pi * 25 * n / 100) for n in range(400)]
    assert rangayyan_eeg_rhythm_detect(x, 100.0)["present"] is False
    r = rangayyan_eeg_rhythm_detect(x, 100.0, band=(20.0, 30.0))
    assert r["present"] is True and r["peak_lag"] == 4
    # the ACF of a 25 Hz rhythm also peaks at lags 8 and 12, inside the
    # alpha lag range; only the fundamental (lag 4) may be used
    assert rangayyan_eeg_rhythm_detect(x, 100.0)["frequency_hz"] == 25.0
    mixed = [math.sin(2 * math.pi * 10 * n / 100) + 0.2 * math.sin(2 * math.pi * 25 * n / 100)
             for n in range(400)]
    assert rangayyan_eeg_rhythm_detect(mixed, 100.0)["peak_lag"] == 10
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_eeg_rhythm_detect(x, 20.0)


