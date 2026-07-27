"""Tests for volsk."""

import numpy as np
import pytest

from morie.fn.volsk import vol_stochastic_kalman


def test_volsk_basic():
    rng = np.random.default_rng(42)
    r = np.concatenate([rng.normal(scale=0.01, size=200), rng.normal(scale=0.05, size=200)])
    out = vol_stochastic_kalman(r)
    assert out["sigma"][250:].mean() > 2 * out["sigma"][:150].mean()


def test_volsk_edge():
    with pytest.raises(ValueError):
        vol_stochastic_kalman(np.ones(5))
    with pytest.raises(ValueError):
        vol_stochastic_kalman(np.ones(20), sigma_eta=0.0)
