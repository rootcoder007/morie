"""Tests for ksr055 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr055 import kosorok_ch2_m_estimator_taylor_expansion


def test_ksr055_basic():
    rng = np.random.default_rng(14)
    X = rng.standard_normal(300)
    m = lambda th, x: (x - th[0]) ** 2
    thetas = [np.array([0.5]), np.array([0.1]), np.array([0.02])]
    out = kosorok_ch2_m_estimator_taylor_expansion(m, thetas, np.array([0.0]), X)
    assert out["ratios"].max() < 1.5


def test_ksr055_edge():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError):
        kosorok_ch2_m_estimator_taylor_expansion(lambda th, x: (x - th[0]) ** 2,
                                                 [np.array([0.1])], np.array([0.0]),
                                                 rng.standard_normal(50))
