"""Tests for ksr050 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr050 import kosorok_ch2_frechet_differentiability


def test_ksr050_basic():
    hs = [np.array(0.1), np.array(0.05), np.array(0.01)]
    assert kosorok_ch2_frechet_differentiability(lambda th: th**2, 1.0,
                                                 hs)["ratio_shrinking"] is True


def test_ksr050_edge():
    hs = [np.array(0.1), np.array(0.05), np.array(0.01)]
    # |.| at 0 is Hadamard but NOT Frechet: the ratio stays at ~1
    kink = kosorok_ch2_frechet_differentiability(lambda th: abs(th), 0.0, hs)
    assert kink["ratios"].min() > 0.9
