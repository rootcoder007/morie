"""Tests for ksr020 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr020 import kosorok_ch1_linear_regression_model


def test_ksr020_basic():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((200, 2))
    Y = Z @ np.array([1.5, -0.5]) + rng.standard_normal(200) * 0.4
    out = kosorok_ch1_linear_regression_model(Y, Z)
    assert out["beta"] == pytest.approx([1.5, -0.5], abs=0.15)
    assert out["bounded_cond_var"] is True


def test_ksr020_edge():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((200, 2))
    with pytest.raises(ValueError):
        kosorok_ch1_linear_regression_model(np.zeros(10), Z)  # length mismatch
