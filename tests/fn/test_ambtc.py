"""Tests for ambtc."""

import numpy as np
import pytest

from morie.fn.ambtc import am_bootstrap_se


def _Z(seed=42, n=60, q=5):
    rng = np.random.default_rng(seed)
    s = np.linspace(-1, 1, q)
    a = rng.normal(scale=0.4, size=n)
    b = rng.uniform(0.6, 1.4, size=n)
    return a[:, None] + b[:, None] * s[None, :] + rng.normal(scale=0.1, size=(n, q))


def test_ambtc_basic():
    out = am_bootstrap_se(_Z(), B=30, seed=0)
    assert out["se"].shape == (5,)
    assert np.all(out["se"] > 0)


def test_ambtc_edge():
    with pytest.raises(ValueError):
        am_bootstrap_se(_Z(), B=5)
    with pytest.raises(ValueError):
        am_bootstrap_se(np.ones(5), B=30)  # 1-D input
