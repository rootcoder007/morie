"""Tests for ksr063 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr063 import kosorok_ch3_cox_efficient_score_beta


def test_ksr063_basic():
    rng = np.random.default_rng(19)
    Z = rng.standard_normal((200, 1))
    T = rng.exponential(1.0 / np.exp(Z[:, 0] * 0.5))
    C = rng.exponential(2.0, 200)
    out = kosorok_ch3_cox_efficient_score_beta(Z, time=np.minimum(T, C),
                                               event=(T <= C).astype(float))
    assert out["efficient_information"][0, 0] > 0


def test_ksr063_edge():
    rng = np.random.default_rng(19)
    Z = rng.standard_normal((50, 1))
    with pytest.raises(ValueError):
        kosorok_ch3_cox_efficient_score_beta(Z, time=np.ones(50), event=None)
