"""Tests for ksr033 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr033 import kosorok_ch2_uniform_covering_number


def test_ksr033_basic():
    rng = np.random.default_rng(8)
    X = rng.random(40)
    F = [(lambda x, c=c: (np.asarray(x) <= c).astype(float)) for c in np.linspace(.1,.9,8)]
    out = kosorok_ch2_uniform_covering_number(F, X, eps=0.1, rng=rng)
    assert out["is_lower_bound"] is True  # sup over Q is sampled


def test_ksr033_edge():
    rng = np.random.default_rng(8)
    X = rng.random(40)
    F = [(lambda x: (np.asarray(x) <= 0.5).astype(float))]
    with pytest.raises(ValueError):
        kosorok_ch2_uniform_covering_number(F, X, eps=1.5)
