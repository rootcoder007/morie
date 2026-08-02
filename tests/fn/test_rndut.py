"""Tests for rndut."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rndut import random_utility_model


def test_rndut_basic():
    V = np.array([1.0, 0.0, -1.0])
    out = random_utility_model(V, "gumbel")
    ez = np.exp(V)
    assert out["probabilities"] == pytest.approx(ez / ez.sum())
    assert out["chosen"] == 0


def test_rndut_edge():
    p = random_utility_model([0.0, 0.0], "normal", n_draws=20000, seed=0)
    assert p["probabilities"][0] == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError):
        random_utility_model([1.0, 2.0], "cauchy")
