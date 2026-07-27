"""Tests for volnsig."""

import numpy as np
import pytest

from morie.fn.volnsig import vol_nelson_skew_garch


def _sim(n=900, seed=0):
    rng = np.random.default_rng(seed)
    e = np.empty(n)
    s2 = 1.0
    for t in range(n):
        if t:
            s2 = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * s2
        e[t] = np.sqrt(s2) * rng.standard_normal()
    return e


def test_volnsig_basic():
    out = vol_nelson_skew_garch(_sim())
    assert np.all(out["sigma2"] > 0)
    assert out["lambda_skew"] > 0
    # the two asymmetries are reported separately
    assert "gamma" in out["params"] and "lambda_skew" in out


def test_volnsig_edge():
    # symmetric data: the skew model cannot beat the symmetric one much
    out = vol_nelson_skew_garch(_sim(seed=2))
    assert out["skew_loglik"] >= out["symmetric_loglik"] - 1e-6
    with pytest.raises(ValueError):
        vol_nelson_skew_garch(np.ones(100))
