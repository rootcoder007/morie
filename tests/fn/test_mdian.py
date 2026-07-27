"""Tests for mdian.mediation_analysis."""

import numpy as np
import pytest

from morie.fn.mdian import mediation_analysis


def test_mdian_basic():
    rng = np.random.default_rng(42)
    n = 1500
    x = rng.normal(size=n)
    m = 0.8 * x + rng.normal(scale=0.7, size=n)
    y = 0.7 * x + 1.5 * m + rng.normal(scale=0.7, size=n)
    out = mediation_analysis(y, x, m)
    assert out["indirect"] == pytest.approx(1.2, abs=0.15)
    assert out["c"] == pytest.approx(out["c_prime"] + out["indirect"], abs=1e-6)


def test_mdian_edge():
    # covariates are residualised out before the paths are fitted, so a
    # covariate that drives both x and m must not inflate the indirect path
    rng = np.random.default_rng(0)
    n = 2000
    cov = rng.normal(size=n)
    x = 0.9 * cov + rng.normal(scale=0.6, size=n)
    m = 0.8 * x + 0.9 * cov + rng.normal(scale=0.6, size=n)
    y = 0.7 * x + 1.5 * m + 0.9 * cov + rng.normal(scale=0.6, size=n)
    adj = mediation_analysis(y, x, m, X=cov)
    naive = mediation_analysis(y, x, m)
    assert abs(adj["indirect"] - 1.2) < abs(naive["indirect"] - 1.2)
