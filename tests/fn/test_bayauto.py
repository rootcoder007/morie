"""bayauto: MCMC autocorrelation and ESS (Geyer 1992)."""

import numpy as np
import pytest

from morie.fn.bayauto import autocorrelation_check as ess


def test_bayauto_iid_draws_have_ess_near_n():
    """Independent draws carry no autocorrelation, so N_eff ~ N."""
    rng = np.random.default_rng(1009)
    r = ess(rng.standard_normal(20_000))
    assert r["ess"] == pytest.approx(20_000, rel=0.15)
    assert r["efficiency"] == pytest.approx(1.0, rel=0.15)


def test_bayauto_ar1_matches_the_analytic_ess():
    """For AR(1) with parameter phi, sum_k rho_k = phi/(1-phi), so

        N_eff = N (1-phi)/(1+phi).

    Ground truth is analytic, so this checks correctness rather than
    self-consistency.
    """
    rng = np.random.default_rng(1013)
    n, phi = 200_000, 0.8
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    expected = n * (1 - phi) / (1 + phi)
    assert ess(x)["ess"] == pytest.approx(expected, rel=0.2)


def test_bayauto_acf_of_ar1_decays_geometrically():
    """rho_k = phi^k."""
    rng = np.random.default_rng(1019)
    n, phi = 100_000, 0.6
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    acf = ess(x)["acf"]
    assert acf[0] == 1.0
    for k in range(1, min(5, len(acf))):
        assert acf[k] == pytest.approx(phi**k, abs=0.03)


def test_bayauto_stickier_chains_have_lower_ess():
    """ESS must fall monotonically as the AR(1) parameter rises."""
    rng = np.random.default_rng(1021)
    n = 40_000
    out = []
    for phi in (0.0, 0.5, 0.9):
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.standard_normal()
        out.append(ess(x)["ess"])
    assert out[0] > out[1] > out[2]


def test_bayauto_truncation_lag_is_finite_and_reported():
    """Geyer's rule must stop well short of N-1, or the estimator is just noise."""
    rng = np.random.default_rng(1031)
    r = ess(rng.standard_normal(5000))
    assert 1 <= r["truncation_lag"] < 5000 - 1
    assert len(r["acf"]) == r["truncation_lag"] + 1


def test_bayauto_antithetic_chain_can_exceed_n():
    """Negative autocorrelation genuinely beats independent sampling.

    A clipped implementation would cap this at N and hide a real property.

    AR(1) with phi = -0.5 gives N_eff = N(1-phi)/(1+phi) = 3N. Note that
    merely negating alternate draws of an iid sequence does NOT do this: a
    symmetric iid sequence stays iid under that map, rho_1 is still ~0, and
    the ESS is still ~N. The autocorrelation has to be built into the
    recursion.
    """
    rng = np.random.default_rng(1033)
    n, phi = 40_000, -0.5
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    got = ess(x)["ess"]
    assert got > n
    assert got == pytest.approx(n * (1 - phi) / (1 + phi), rel=0.2)


def test_bayauto_constant_chain_is_a_failure_not_zero():
    """A chain that never moved has an undefined ESS; report it."""
    with pytest.raises(ValueError, match="never moved"):
        ess(np.full(100, 3.0))


def test_bayauto_rejects_too_few_draws():
    with pytest.raises(ValueError, match="at least 4 draws"):
        ess(np.array([1.0, 2.0, 3.0]))
