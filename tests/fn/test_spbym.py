"""spbym -- Besag-York-Mollie convolution, BYM (1991) Sec. 4."""

import numpy as np
import pytest


def _areas(n=10, seed=3):
    """A chain with two extra links, so degrees vary."""
    rs = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    A[0, 4] = A[4, 0] = 1.0
    E = rs.uniform(8, 30, n)
    y = np.array([float(rs.poisson(e * np.exp(x)))
                  for e, x in zip(E, np.linspace(-0.3, 0.3, n))])
    return A, E, y

from morie.fn.spbym import schabenberger_bym

KAPPA, LAM = 0.129, 0.011          # the paper's thyroid-cancer estimates


def test_sum_of_v_is_zero():
    """Stated in Sec. 4; here a consequence of stationarity."""
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert abs(r["sum_v"]) < 1e-8


def test_fitted_total_matches_the_observed_total():
    """sum c_i exp(u*_i + v*_i) = sum y_i, Sec. 4."""
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert r["fitted_total"] == pytest.approx(r["observed_total"], abs=1e-7)
    assert r["observed_total"] == pytest.approx(float(y.sum()))


def test_converges_without_warnings():
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert r["converged"]
    assert "warning" not in r


def test_relative_risk_is_exp_of_the_convolution():
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert np.allclose(r["x"], r["u"] + r["v"])
    assert np.allclose(r["relative_risk"], np.exp(r["x"]))


def test_kappa_towards_zero_forces_u_constant():
    """Sec. 4's reading of the scale parameters."""
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, 1e-6, LAM)
    assert float(np.ptp(r["u"])) < 1e-3


def test_lambda_towards_zero_forces_v_to_vanish():
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, 1e-8)
    assert float(np.max(np.abs(r["v"]))) < 1e-5


def test_smr_is_reported_alongside_the_smoothed_risk():
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert np.allclose(r["smr"], y / E)


def test_identifiability_is_stated():
    A, E, y = _areas()
    r = schabenberger_bym(y, E, A, KAPPA, LAM)
    assert "not separately identifiable" in r["identifiability"]


def test_bad_inputs_rejected():
    A, E, y = _areas()
    with pytest.raises(ValueError):
        schabenberger_bym(y, -E, A, KAPPA, LAM)
    with pytest.raises(ValueError):
        schabenberger_bym(y, E, A, -1.0, LAM)
