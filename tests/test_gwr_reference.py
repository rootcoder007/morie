"""Basic GWR against GWmodel::gwr.basic (coefficients, SEs, local R2, AICc,
enp to 1e-13 for fixed bisquare, adaptive Gaussian and adaptive tricube);
tests/cross/test-morie_vs_GWmodel.R repeats that in R.
"""

import math

import pytest

from morie.fn.gwrbas import gwr_basic, gwr_kernel_weights


def _data():
    P = [(x * 1.0, y * 1.0) for x in range(4) for y in range(4)]
    X = [[1.0, 0.3 * i % 1.7] for i in range(16)]
    y = [1.0 + 2.0 * r[1] + 0.1 * p[0] for r, p in zip(X, P)]
    return y, X, P


def test_kernel_weights():
    assert [round(w, 6) for w in gwr_kernel_weights([0.0, 1.0, 2.0, 3.0], 2.5)] == [1.0, 0.7056, 0.1296, 0.0]
    g = gwr_kernel_weights([0.0, 1.0, 2.0], 2.0, "gaussian")
    assert g == pytest.approx([1.0, math.exp(-0.125), math.exp(-0.5)], abs=1e-15)
    # adaptive: the 2nd smallest distance (self included) is the bandwidth
    assert gwr_kernel_weights([0.0, 1.0, 2.0, 3.0], 2, "boxcar", adaptive=True) == [1.0, 1.0, 0.0, 0.0]


def test_huge_bandwidth_reproduces_ols():
    y, X, P = _data()
    r = gwr_basic(y, X, P, 1e6, kernel="gaussian")
    # all weights ~1: every local fit is the global OLS fit
    b0 = r.betas[0]
    assert all(row == pytest.approx(b0, abs=1e-6) for row in r.betas)
    assert r.diagnostics["trS"] == pytest.approx(2.0, abs=1e-6)


def test_documented_aicc_and_hat_identities():
    y, X, P = _data()
    r = gwr_basic(y, X, P, 3.0, kernel="gaussian")
    assert round(r.diagnostics["AICc"], 6) == -20.824713
    d = r.diagnostics
    assert d["edf"] + d["enp"] == pytest.approx(16.0, abs=1e-12)
    assert sum(v * v for v in r.residuals) == pytest.approx(d["RSS"], abs=1e-15)
