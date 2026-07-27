"""Tests for pscli."""

import numpy as np
import pytest

from scipy.stats import norm

from morie.fn.pscli import pscl_ideal
from morie.fn.pscrc import pscl_rollcall


def test_pscli_basic():
    rng = np.random.default_rng(42)
    n, q = 30, 50
    x = np.linspace(-2, 2, n)
    beta = rng.normal(size=q)
    V = (rng.random((n, q)) < norm.cdf(x[:, None] * beta)).astype(float)
    V = np.column_stack([V, np.ones(n)])  # one unanimous roll call
    rc = pscl_rollcall(V)
    out = pscl_ideal(rc, n_iter=250, burnin=100, seed=0, polarity_idx=0)
    assert out["n_rollcalls_dropped"] == 1
    assert abs(np.corrcoef(out["ideal_points"], x)[0, 1]) > 0.85


def test_pscli_edge():
    with pytest.raises(ValueError):
        pscl_ideal({"votes": np.ones((4, 3))})  # missing keep
