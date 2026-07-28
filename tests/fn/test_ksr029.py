"""Tests for ksr029 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr029 import kosorok_ch2_glivenko_cantelli_class


def test_ksr029_basic():
    rng = np.random.default_rng(5)
    F = [(lambda x, c=c: (np.asarray(x) <= c).astype(float)) for c in (0.3, 0.6)]
    assert kosorok_ch2_glivenko_cantelli_class(F, rng.random(2000))["shrinking"] is True


def test_ksr029_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_glivenko_cantelli_class([], np.random.default_rng(5).random(50))
