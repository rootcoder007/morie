"""Tests for npstm."""

from morie.fn import _array_core as np
import pytest

from morie.fn.npstm import nonparametric_tmle_survival


def test_npstm_basic():
    rng = np.random.default_rng(42)
    n = 2000
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 1 / (1 + np.exp(-W[:, 0]))).astype(float)
    t_event = rng.exponential(np.exp(0.6 * A))
    cens = rng.exponential(4.0, size=n)
    time = np.minimum(t_event, cens)
    event = (t_event <= cens).astype(float)
    out = nonparametric_tmle_survival(time, event, A, W)
    assert out["rmst_difference"] > 0
    assert out["rmst1"] > out["rmst0"]


def test_npstm_edge():
    rng = np.random.default_rng(0)
    n = 500
    W = rng.normal(size=(n, 2))
    A = (rng.random(n) < 0.5).astype(float)
    time = rng.exponential(size=n)
    event = np.ones(n)
    with pytest.raises(ValueError):
        nonparametric_tmle_survival(-time, event, A, W)  # nonpositive times
