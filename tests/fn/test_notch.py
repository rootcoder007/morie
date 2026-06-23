"""Test notch_filter_signal (notch)."""

import numpy as np

from morie.fn._containers import SignalResult
from morie.fn.notch import notch, notch_filter_signal


class TestNotchFilter:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = notch_filter_signal(x, freq=60.0, fs=1000.0)
        assert isinstance(result, SignalResult)
        assert result.name == "notch_filter_signal"

    def test_output_shape(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = notch_filter_signal(x, freq=60.0, fs=1000.0)
        assert result.n_samples == 256

    def test_fs_stored(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = notch_filter_signal(x, freq=60.0, fs=1000.0)
        assert result.fs == 1000.0

    def test_q_param(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = notch_filter_signal(x, freq=50.0, fs=500.0, q=20.0)
        assert isinstance(result, SignalResult)

    def test_alias(self):
        assert notch is notch_filter_signal
