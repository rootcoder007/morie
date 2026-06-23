"""Tests for morie.fn.ssim1 -- SSIM."""

import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.ssim1 import ssim, ssim1


class TestSsim1:
    def test_alias(self):
        assert ssim1 is ssim

    def test_identical(self):
        x = np.random.default_rng(42).random(100)
        result = ssim(x, x, win_size=7)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.99

    def test_different(self):
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)
        result = ssim(x, y)
        assert result.value < 0.5
