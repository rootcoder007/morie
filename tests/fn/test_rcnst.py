"""Test sinc_reconstruct (rcnst)."""

import numpy as np

from morie.fn._containers import SignalResult
from morie.fn.rcnst import rcnst, sinc_reconstruct


class TestSincReconstruct:
    def test_basic(self):
        fs = 100.0
        t = np.arange(50) / fs
        samples = np.sin(2 * np.pi * 5 * t)
        t_new = np.linspace(0, t[-1], 200)
        result = sinc_reconstruct(samples, fs, t_new)
        assert isinstance(result, SignalResult)
        assert result.name == "sinc_reconstruct"
        assert result.n_samples == 200

    def test_exact_at_samples(self):
        fs = 100.0
        t = np.arange(20) / fs
        samples = np.sin(2 * np.pi * 5 * t)
        result = sinc_reconstruct(samples, fs, t)
        np.testing.assert_allclose(result.filtered, samples, atol=1e-10)

    def test_alias(self):
        assert rcnst is sinc_reconstruct
