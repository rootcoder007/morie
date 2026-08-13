"""Tests for abcgp -- ABC with a Gaussian-process surrogate.

The generated tests these replace called the fabricated stub, which
returned mean(obs) with a standard error. Every assertion here is
against a closed form, a printed property of the Sobol sequence, or a
limiting case; the full set lives in the anchor,
ledger/wave3/anchor_abcgp.py.
"""

import math

import pytest

from morie.fn.abcgp import (abc_gp_emulator, design_from_prior,
                            gabc_log_likelihood, gp_fit, gp_predict,
                            gps_abc, implausible, sobol_sequence,
                            synthetic_abc, synthetic_log_likelihood)

S2 = 0.25
EPS = 0.8
D_OBS = 1.0


def gauss_sim(theta, rng):
    """X ~ N(theta, S2), one observation per call."""
    return [theta[0] + math.sqrt(S2) * rng.standard_normal()]


def closed_form_log_gabc(theta, eps=EPS):
    """E_X[exp(-(X-D)^2 / 2 eps^2)] for X ~ N(theta, S2), in closed form."""
    v = eps ** 2 + S2
    return 0.5 * math.log(eps ** 2 / v) - (D_OBS - theta) ** 2 / (2.0 * v)


# ------------------------------------------------------------ Sobol
def test_sobol_first_points_are_a_net():
    """The first 2^m points are a (0, m, s)-net in base 2.

    One point in every elementary box of volume 2^-m, for every box
    shape. This is the defining property and a wrong direction-number
    table breaks it.
    """
    pts = sobol_sequence(8, 2).tolist()
    for kx, ky in ((8, 1), (4, 2), (2, 4), (1, 8)):
        counts = {}
        for x, y in pts:
            cell = (int(x * kx), int(y * ky))
            counts[cell] = counts.get(cell, 0) + 1
        assert sorted(counts.values()) == [1] * 8, (kx, ky)


def test_sobol_one_dimension_is_gray_code_radical_inverse():
    one_d = [round(r[0], 6) for r in sobol_sequence(8, 1).tolist()]
    assert one_d == [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]
    assert sorted(one_d) == [i / 8.0 for i in range(8)]


def test_sobol_rejects_untabulated_dimensions():
    with pytest.raises(ValueError):
        sobol_sequence(4, 99)
    with pytest.raises(ValueError):
        sobol_sequence(0, 2)


def test_design_maps_onto_the_prior_box():
    d0 = design_from_prior(4, ([-2.0, 0.0], [2.0, 10.0]), skip=0).tolist()
    d1 = design_from_prior(4, ([-2.0, 0.0], [2.0, 10.0])).tolist()
    assert all(-2.0 <= r[0] <= 2.0 and 0.0 <= r[1] <= 10.0
               for r in d0 + d1)
    # skip=0 is the box corner; the default skips it
    assert d0[0] == pytest.approx([-2.0, 0.0])
    assert d1[0] == pytest.approx([0.0, 5.0])


def test_design_through_quantile_functions():
    """Wilkinson Sec. 2.2's non-uniform case: apply the inverse CDF."""
    d = design_from_prior(6, [lambda u: -math.log(1.0 - u)]).tolist()
    assert all(r[0] > 0.0 for r in d)
    assert sorted(r[0] for r in d) == pytest.approx(
        sorted(-math.log(1.0 - u[0])
               for u in sobol_sequence(6, 1, skip=1).tolist()))


# --------------------------------------------------- the likelihood
def test_eq1_converges_to_the_closed_form():
    """Wilkinson eq. (1) is an unbiased estimator of a known integral."""
    for th in (0.0, 0.5, 1.0, 1.5):
        got, _ = gabc_log_likelihood(gauss_sim, [D_OBS], [th], n_sim=20000,
                                     epsilon=EPS, seed=11)
        assert got == pytest.approx(closed_form_log_gabc(th), abs=0.02)


def test_nugget_is_the_sampling_variance():
    """v^2 must fall like 1/M -- it is the variance of the estimate."""
    _, v_small = gabc_log_likelihood(gauss_sim, [D_OBS], [1.0], n_sim=500,
                                     epsilon=EPS, seed=3, bootstrap=200)
    _, v_big = gabc_log_likelihood(gauss_sim, [D_OBS], [1.0], n_sim=2000,
                                   epsilon=EPS, seed=3, bootstrap=200)
    assert 2.0 < v_small / v_big < 8.0


def test_uniform_kernel_is_rejection_abc():
    """pi(D|X) = 1{rho <= eps} makes eq. (1) the acceptance rate."""
    got, _ = gabc_log_likelihood(gauss_sim, [D_OBS], [1.0], n_sim=20000,
                                 epsilon=EPS, kernel="uniform", seed=5)
    want = math.erf(EPS / math.sqrt(2.0 * S2))
    assert math.exp(got) == pytest.approx(want, abs=0.01)


def test_rejects_bad_kernel_and_tolerance():
    with pytest.raises(ValueError):
        gabc_log_likelihood(gauss_sim, [D_OBS], [1.0], kernel="nope")
    with pytest.raises(ValueError):
        gabc_log_likelihood(gauss_sim, [D_OBS], [1.0], epsilon=0.0)


def test_synthetic_likelihood_is_the_normal_density():
    draws = [[x] for x in (0.5, 1.0, 1.5, 2.0, 2.5)]
    ll, mu, cov = synthetic_log_likelihood(draws, [1.0], epsilon=0.0)
    m = 1.5
    s2 = sum((x[0] - m) ** 2 for x in draws) / 4.0
    want = -0.5 * (math.log(2.0 * math.pi * s2) + (1.0 - m) ** 2 / s2)
    assert ll == pytest.approx(want, abs=1e-12)
    assert mu[0] == pytest.approx(m)
    # eq. (9): eps^2 goes on the diagonal
    _, _, cov_e = synthetic_log_likelihood(draws, [1.0], epsilon=0.5)
    assert cov_e[0][0] == pytest.approx(cov[0][0] + 0.25)


def test_synthetic_likelihood_needs_a_covariance():
    with pytest.raises(ValueError):
        synthetic_log_likelihood([[1.0]], [1.0])


# ------------------------------------------------------ the emulator
def _quadratic_fit():
    grid = [[t] for t in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 0.5, -1.5)]
    vals = [closed_form_log_gabc(t[0]) for t in grid]
    return grid, vals, gp_fit(grid, vals, nugget=1e-10)


def test_gp_interpolates_and_extrapolates_a_quadratic():
    """The prior mean of Sec. 2.1 is quadratic, and so is this target."""
    grid, vals, fit = _quadratic_fit()
    m0, s0 = gp_predict(fit, grid[3])
    assert m0 == pytest.approx(vals[3], abs=1e-6)
    assert s0 < 1e-3
    for t in (0.25, 0.75, 1.25, 1.75):
        assert gp_predict(fit, [t])[0] == pytest.approx(
            closed_form_log_gabc(t), abs=1e-6)


def test_gp_uncertainty_grows_away_from_the_design():
    """If it did not, eq. (3)'s m + 3 sigma would be meaningless."""
    _, _, fit = _quadratic_fit()
    _, s_in = gp_predict(fit, [0.5])
    _, s_out = gp_predict(fit, [12.0])
    assert s_out > 10.0 * s_in


def test_gp_rejects_an_underdetermined_design():
    with pytest.raises(ValueError):
        gp_fit([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0])   # 3 pts, 3 coefs
    with pytest.raises(ValueError):
        gp_fit([[0.0], [1.0]], [1.0, 2.0])


# ------------------------------------------------- history matching
def test_implausibility_rule():
    """Eq. (3), with Wilkinson's T = 10."""
    grid, vals, fit = _quadratic_fit()
    best = grid[max(range(len(vals)), key=lambda i: vals[i])]
    assert not implausible(fit, best)
    assert implausible(fit, [8.0])
    # T controls it; nothing is implausible against an enormous threshold
    assert not implausible(fit, [8.0], threshold=1e6)


def test_history_matching_rules_out_space():
    """Waves must actually shrink the space, or Sec. 3 is switched off.

    This is the check that caught a real bug: gp_predict formed the
    integrated-mean variance correction against A^-1 H instead of H, so
    sigma came out around 10 log-units at visited design points and
    m + 3 sigma cleared the threshold everywhere.
    """
    r = abc_gp_emulator(gauss_sim, [D_OBS],
                        X_grid=[[t / 20.0] for t in range(-40, 61)],
                        prior_ppf=([-3.0], [3.0]), n_waves=3, n_design=14,
                        n_sim=400, epsilon=0.5, seed=17)
    assert sum(w["ruled_implausible"] for w in r["waves"]) > 0
    # the emulator models the LIKELIHOOD, so its peak is the MLE
    assert r["estimate"][0] == pytest.approx(D_OBS, abs=0.15)
    assert len(r["posterior"]) == len(r["grid"])
    assert sum(r["posterior"]) == pytest.approx(1.0)


# ------------------------------------------------- Meeds & Welling
def log_prior(theta):
    return -0.5 * theta[0] ** 2 / 4.0


def test_gps_abc_recovers_the_conjugate_posterior():
    """N(theta, S2) data with an N(0, 4) prior has a known posterior."""
    post_v = 1.0 / (1.0 / S2 + 1.0 / 4.0)
    post_m = post_v * (D_OBS / S2)
    g = gps_abc(gauss_sim, [D_OBS], log_prior, [0.0], n_iter=400, n_sim=8,
                epsilon=0.0, proposal_sd=0.6, seed=23, xi=0.1, delta_s=8,
                n_alpha=48, max_sim=64)
    kept = g["chain"][len(g["chain"]) // 2:]
    mean = sum(r[0] for r in kept) / len(kept)
    assert mean == pytest.approx(post_m, abs=0.15)
    assert 0.05 < g["acceptance_rate"] < 0.95


def test_synthetic_abc_agrees_with_gps_abc():
    """Algorithm 1 and Algorithm 2 target the same posterior."""
    post_v = 1.0 / (1.0 / S2 + 1.0 / 4.0)
    post_m = post_v * (D_OBS / S2)
    s = synthetic_abc(gauss_sim, [D_OBS], log_prior, [0.0], n_iter=400,
                      n_sim=16, epsilon=0.0, proposal_sd=0.6, seed=29)
    kept = s["chain"][len(s["chain"]) // 2:]
    mean = sum(r[0] for r in kept) / len(kept)
    assert mean == pytest.approx(post_m, abs=0.2)


def test_front_end_argument_checks():
    with pytest.raises(ValueError):
        abc_gp_emulator(gauss_sim, [D_OBS], method="nope")
    with pytest.raises(ValueError):
        abc_gp_emulator("not a callable", [D_OBS])
    with pytest.raises(ValueError):
        abc_gp_emulator(gauss_sim, [D_OBS])              # no prior_ppf
    with pytest.raises(ValueError):
        abc_gp_emulator(gauss_sim, [D_OBS], method="gps")  # no theta0
