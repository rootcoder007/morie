"""Tests for wsmiis.wasserman_importance_sampling (MacKay 29.21-29.22)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.wsmiis import wasserman_importance_sampling


def test_wsmiis_basic():
    """w_r = P*(x_r)/Q*(x_r); the self-normalised estimate is
    sum w phi / sum w; ESS = (sum w)^2 / sum w^2; the unnormalised one
    is mean(w phi)."""
    xs = [-2.0 + 4.0 * k / 49 for k in range(50)]
    # the callables are evaluated on the whole array of draws
    p = lambda x: np.exp(-0.5 * (np.asarray(x) - 0.3) ** 2)
    q = lambda x: np.exp(-0.5 * np.asarray(x) ** 2 / 4)
    f = lambda x: np.asarray(x) ** 2
    w = [math.exp(-0.5 * (x - 0.3) ** 2) / math.exp(-0.5 * x * x / 4) for x in xs]
    r = wasserman_importance_sampling(f, p, q, samples=xs)
    assert list(r["weights"]) == pytest.approx(w, rel=1e-14)
    assert r["estimate"] == pytest.approx(sum(a * x * x for a, x in zip(w, xs)) / sum(w), rel=1e-13)
    assert r["effective_sample_size"] == pytest.approx(sum(w) ** 2 / sum(a * a for a in w), rel=1e-13)
    ru = wasserman_importance_sampling(f, p, q, samples=xs, normalised=True)
    assert ru["estimate"] == pytest.approx(sum(a * x * x for a, x in zip(w, xs)) / 50, rel=1e-13)


def test_wsmiis_edge():
    """Q = P gives equal weights and the plain sample mean; no samples
    and no sampler raise."""
    xs = [0.1, 0.5, 0.9]
    one = lambda x: np.ones(len(x))
    r = wasserman_importance_sampling(lambda x: np.asarray(x), one, one, samples=xs)
    assert r["estimate"] == pytest.approx(0.5, abs=1e-15)
    assert r["effective_sample_size"] == pytest.approx(3.0, abs=1e-12)
    with pytest.raises(ValueError):
        wasserman_importance_sampling(lambda x: x, one, one)


