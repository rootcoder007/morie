"""Tests for volmsg."""

import numpy as np
import pytest

from morie.fn.volmsg import vol_markov_switching_garch


def test_volmsg_basic():
    rng = np.random.default_rng(0)
    n = 800
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.98 else 1 - state[t - 1]
    r = rng.standard_normal(n) * np.where(state == 1, 2.0, 0.5)
    out = vol_markov_switching_garch(r)
    assert out["unconditional_var"][0] < out["unconditional_var"][1]  # calm first
    assert np.allclose(out["transition"].sum(axis=1), 1.0)


def test_volmsg_edge():
    with pytest.raises(ValueError):
        vol_markov_switching_garch(np.random.default_rng(1).normal(size=500), K=1)
    with pytest.raises(ValueError):
        vol_markov_switching_garch(np.zeros(50))  # too short
