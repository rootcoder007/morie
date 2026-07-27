"""Tests for garchm."""

import numpy as np
import pytest

from morie.fn.garchm import garch_model


def _sim(n=900, seed=0):
    """GARCH(1,1) path: omega 0.05, alpha 0.1, beta 0.85 (uncond. var 1)."""
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    s2 = 1.0
    for t in range(n):
        if t:
            s2 = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * s2
        e[t] = np.sqrt(s2) * rng.standard_normal()
    return e


def test_garchm_basic():
    out = garch_model(_sim())
    assert np.all(out["sigma2"] > 0)
    assert np.isfinite(out["loglik"])
    assert out["spec"] == "garch"


def test_garchm_edge():
    with pytest.raises(ValueError):
        garch_model(np.arange(10.0))  # too short to fit
    with pytest.raises(ValueError):
        garch_model(np.ones(100))  # zero variance
