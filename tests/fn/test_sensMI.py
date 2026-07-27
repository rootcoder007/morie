"""Tests for sensMI.sensitivity_mediation_imbens.

Checked against the identities in Imai, Keele & Yamamoto (2010),
*Statistical Science* 25(1), 51-71, rather than against a reference
implementation: Theorem 2 (ACME = beta2 * gamma), Theorem 4 (the ACME as
a function of rho), and the remark that the ACME vanishes exactly at
rho == rho-tilde.
"""

import numpy as np
import pytest

from morie.fn.sensMI import sensitivity_mediation_imbens


def _mediation_data(seed=11, n=400, b2=0.8, g=0.5, b3=0.3):
    """T -> M -> Y with a known indirect path, so beta2*gamma is known."""
    rng = np.random.default_rng(seed)
    T = rng.integers(0, 2, n).astype(float)
    M = 1.0 + b2 * T + rng.normal(0, 1, n)
    Y = 2.0 + b3 * T + g * M + rng.normal(0, 1, n)
    return Y, T, M


def test_recovers_the_indirect_path():
    Y, T, M = _mediation_data()
    res = sensitivity_mediation_imbens(Y, T, M)
    assert res["beta2"] == pytest.approx(0.8, abs=0.15)
    assert res["gamma"] == pytest.approx(0.5, abs=0.10)
    assert res["estimate"] == pytest.approx(0.8 * 0.5, abs=0.12)


def test_theorem_4_reduces_to_theorem_2_at_rho_zero():
    """At rho = 0 the two identifications must agree exactly, not roughly.

    OLS makes e3 orthogonal to e2 in sample and e1 = gamma*e2 + e3, so
    rho-tilde * sigma1 / sigma2 == gamma identically. Theorem 4's leading
    term beta2*sigma1*rho-tilde/sigma2 is then beta2*gamma to machine
    precision.
    """
    Y, T, M = _mediation_data()
    res = sensitivity_mediation_imbens(Y, T, M, r2_grid=[0.0])
    assert res["acme_positive"][0] == pytest.approx(res["estimate"], rel=1e-10)
    assert res["acme_negative"][0] == pytest.approx(res["estimate"], rel=1e-10)


def test_acme_vanishes_at_the_breakdown_correlation():
    """ACME(rho) == 0 iff rho == rho-tilde (Theorem 4, third remark)."""
    Y, T, M = _mediation_data()
    res = sensitivity_mediation_imbens(Y, T, M)
    rho_t = res["rho_breakdown"]
    at_breakdown = sensitivity_mediation_imbens(Y, T, M, r2_grid=[rho_t**2])
    branch = "acme_positive" if rho_t > 0 else "acme_negative"
    assert at_breakdown[branch][0] == pytest.approx(0.0, abs=1e-10)


def test_acme_is_monotone_in_rho():
    """Theorem 4's second remark: monotone in rho, direction set by beta2."""
    Y, T, M = _mediation_data()
    res = sensitivity_mediation_imbens(Y, T, M, r2_grid=np.linspace(0, 0.9, 12))
    d = np.diff(res["acme_positive"])
    assert np.all(d < 0) or np.all(d > 0)


def test_rho_grid_is_the_square_root_of_the_r2_product():
    """rho^2 = R2*_M R2*_Y, so the grid maps by a square root."""
    Y, T, M = _mediation_data()
    grid = [0.0, 0.25, 0.64]
    res = sensitivity_mediation_imbens(Y, T, M, r2_grid=grid)
    assert res["rho_grid"] == pytest.approx([0.0, 0.5, 0.8])


def test_rejects_multivariate_treatment():
    Y, T, M = _mediation_data()
    with pytest.raises(ValueError, match="one-dimensional"):
        sensitivity_mediation_imbens(Y, np.column_stack([T, T]), M)


def test_rejects_r2_outside_unit_interval():
    Y, T, M = _mediation_data()
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        sensitivity_mediation_imbens(Y, T, M, r2_grid=[1.0])


def test_rejects_length_mismatch():
    Y, T, M = _mediation_data()
    with pytest.raises(ValueError, match="same length"):
        sensitivity_mediation_imbens(Y, T, M[:-1])
