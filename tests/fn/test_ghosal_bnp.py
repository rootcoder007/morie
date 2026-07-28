"""Bayesian nonparametrics (Ghosal & van der Vaart)."""

import numpy as np
import pytest

from morie.fn.gh_c3_14 import ghosal_mpt_prior
from morie.fn.gh_c5_7 import ghosal_pred_rec
from morie.fn.gh_c5_8 import ghosal_gauss_ker
from morie.fn.gh_c5_9 import ghosal_beta_ker
from morie.fn.gh_c7_4 import ghosal_norm_mix_con
from morie.fn.gh_c7_6 import ghosal_pt_dens_con
from morie.fn.gh_c7_7 import ghosal_spec_dens_con
from morie.fn.gh_c8_6 import ghosal_iid_crt_thm
from morie.fn.gh_c10_7 import ghosal_frs_density
from morie.fn.gh_c11_4 import ghosal_gp_dens_crt
from morie.fn.gh_c13_2 import ghosal_surv_dp_km
from morie.fn.gh_c13_15 import ghosal_cox_bvm
from morie.fn.gh_dp_post_ex import ghosal_dp_posterior_exact
from morie.fn.gh_emp_bayes import ghosal_empirical_bayes_np
from morie.fn.gh_pt_adapt import ghosal_pt_adaptive


def test_dp_predictive_is_a_mixture_whose_weights_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    out = ghosal_dp_posterior_exact(x, alpha=2.0)
    assert out["base_weight"] + out["atom_weight"] == pytest.approx(1.0)
    assert out["atom_probs"].sum() == pytest.approx(out["atom_weight"])
    # a DP draw is a.s. discrete: this is NOT a density
    assert out["is_density"] is False
    # alpha -> 0 puts everything on the data, alpha -> inf on G0
    assert ghosal_dp_posterior_exact(x, alpha=1e-6)["atom_weight"] > 0.999
    assert ghosal_dp_posterior_exact(x, alpha=1e6)["base_weight"] > 0.999
    with pytest.raises(ValueError):
        ghosal_dp_posterior_exact(x, alpha=0.0)


def test_predictive_recursion_is_a_density_and_order_dependent():
    rng = np.random.default_rng(1)
    theta = rng.choice([-2.0, 2.0], 200)
    x = theta + rng.standard_normal(200) * 0.7
    out = ghosal_pred_rec(x, sigma=0.7)
    th, f = out["theta_grid"], out["f_mixing"]
    assert np.trapezoid(f, th) == pytest.approx(1.0, abs=1e-6)
    # the mixing law is bimodal near +-2, which a naive density of x
    # would not resolve at this noise level
    assert f[np.argmin(np.abs(th + 2))] > f[np.argmin(np.abs(th))]
    assert f[np.argmin(np.abs(th - 2))] > f[np.argmin(np.abs(th))]
    # the section's own caveat, made checkable
    assert out["order_dependent"] is True
    shuffled = ghosal_pred_rec(rng.permutation(x), sigma=0.7)
    assert not np.allclose(out["f_mixing"], shuffled["f_mixing"])


def test_dp_mixtures_produce_actual_densities():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(100)
    g = ghosal_gauss_ker(x, n_draws=60)
    assert g["is_density"] is True
    assert np.all(g["density"] >= 0)
    assert np.trapezoid(g["density"], g["grid"]) == pytest.approx(1.0, abs=0.15)
    # truncation is reported, not assumed away
    assert 0.0 <= g["truncation_mass"] < 0.05


def test_beta_kernel_keeps_all_its_mass_inside_the_unit_interval():
    rng = np.random.default_rng(3)
    x = rng.beta(2.0, 5.0, 150)
    out = ghosal_beta_ker(x, n_draws=60)
    assert out["support"] == (0.0, 1.0)
    assert out["mass_outside_support"] == 0.0
    assert np.all(out["density"] >= 0)
    # a Gaussian kernel would leak past the boundary; this cannot
    assert np.all((out["grid"] > 0) & (out["grid"] < 1))
    with pytest.raises(ValueError):
        ghosal_beta_ker(rng.standard_normal(50))     # outside [0, 1]


def test_polya_tree_posterior_is_a_density_and_needs_growing_a_m():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(300)
    out = ghosal_pt_dens_con(x, levels=5)
    assert out["absolutely_continuous_prior"] is True
    assert np.all(out["density"] >= 0)
    assert out["mass"] == pytest.approx(1.0, abs=0.2)
    assert "m^2" in out["a_rule"]
    with pytest.raises(ValueError):
        ghosal_pt_dens_con(x, a_scale=0.0)


def test_mixing_polya_trees_smooths_the_partition_artefacts():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(400)
    out = ghosal_mpt_prior(x, levels=5)
    # the mixture must be no rougher than a single tree: that is the
    # entire reason the section exists
    assert out["smoother_than_single"] is True
    assert out["max_jump"] <= out["max_jump_single"]
    assert out["n_components"] > 1


def test_adaptive_polya_tree_pays_exactly_one_log_factor():
    out = ghosal_pt_adaptive(np.zeros(5), s=2.0, n=10_000)
    assert out["adaptive"] is True
    assert out["requires_knowing_s"] is False
    # near-optimal means minimax times log n, and the factor is
    # returned separately so "near" is a number
    assert out["rate"] == pytest.approx(out["minimax_rate"] * np.log(10_000))
    assert out["ratio_to_minimax"] == pytest.approx(np.log(10_000))
    assert out["rate"] > out["minimax_rate"]
    # the same prior serves every smoothness
    assert len(out["scan"]) >= 4
    with pytest.raises(ValueError):
        ghosal_pt_adaptive(np.zeros(5), s=-1.0, n=100)


def test_normal_mixture_consistency_is_measured_against_a_reference():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(200)
    out = ghosal_norm_mix_con(x, n_draws=60)
    assert 0.0 <= out["hellinger_to_reference"] < 0.5
    assert "KL neighbourhood" in out["requires"]


def test_contraction_theorem_balances_entropy_against_prior_mass():
    out = ghosal_iid_crt_thm(np.zeros(9), eps=0.1, n=1000)
    assert out["n_eps_squared"] == pytest.approx(10.0)
    assert out["entropy_budget"] == pytest.approx(10.0)
    assert out["prior_mass_budget"] == pytest.approx(np.exp(-10.0))
    assert out["metric"] == "Hellinger"
    # both conditions are checked only when both are supplied: the
    # theorem needs them TOGETHER
    partial = ghosal_iid_crt_thm(np.zeros(9), eps=0.1, n=1000, entropy=5.0)
    assert partial["entropy_ok"] is True
    assert partial["prior_mass_ok"] is None
    assert partial["all_conditions_checked"] is False
    full = ghosal_iid_crt_thm(np.zeros(9), eps=0.1, n=1000, entropy=5.0,
                             prior_mass=np.exp(-5.0))
    assert full["all_conditions_checked"] is True
    assert full["prior_mass_ok"] is True


def test_squared_exponential_gp_contracts_only_logarithmically():
    n = 10_000
    se = ghosal_gp_dens_crt(np.zeros(5), s=1.0, n=n)
    ma = ghosal_gp_dens_crt(np.zeros(5), s=1.0, n=n, kernel="matern")
    rs = ghosal_gp_dens_crt(np.zeros(5), s=1.0, n=n, kernel="rescaled_se")
    # the point of the section: an analytic-path prior is too smooth
    assert se["rate_kind"] == "LOGARITHMIC"
    assert se["attains_minimax"] is False
    assert se["rate"] > ma["rate"]            # far worse
    assert ma["attains_minimax"] is True
    assert ma["rate"] == pytest.approx(ma["minimax_rate"])
    # A logarithmic rate is bad ASYMPTOTICALLY, not at every n. At
    # n = 1e4 the squared-exponential rate (0.109) is numerically
    # SMALLER than the rescaled polynomial one (0.204); they cross
    # later. What actually diverges is the ratio to minimax:
    # measured 2.3 at 1e4, 7.2 at 1e6, 94 at 1e10.
    ratios = [ghosal_gp_dens_crt(np.zeros(5), s=1.0, n=m)["ratio_to_minimax"]
              for m in (10_000, 1_000_000, 10_000_000_000)]
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[2] > 50
    # and the rescaled prior is polynomial, so ITS ratio stays bounded
    rs_ratios = [ghosal_gp_dens_crt(np.zeros(5), s=1.0, n=m,
                                    kernel="rescaled_se")["ratio_to_minimax"]
                 for m in (10_000, 10_000_000_000)]
    assert rs_ratios[1] < ratios[2]
    with pytest.raises(ValueError):
        ghosal_gp_dens_crt(np.zeros(5), kernel="laplace")


def test_finite_random_series_is_adaptive_through_the_prior_on_K():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(200)
    adaptive = ghosal_frs_density(x, n_draws=40)
    fixed = ghosal_frs_density(x, K=3, n_draws=40)
    assert adaptive["adaptive"] is True
    assert fixed["adaptive"] is False
    assert fixed["K_fixed"] == 3
    assert adaptive["K_drawn_mean"] > 0
    assert np.all(adaptive["density"] >= 0)
    assert adaptive["mass"] == pytest.approx(1.0, abs=0.05)


def test_whittle_likelihood_runs_on_a_dependent_series():
    rng = np.random.default_rng(8)
    n = 256
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + e[t]        # AR(1), genuinely dependent
    out = ghosal_spec_dens_con(y)
    assert out["exact"] is False
    assert np.all(out["spectral_density"] > 0)
    assert np.isfinite(out["whittle_loglik"])
    # an AR(1) with positive coefficient has more power at low
    # frequency: the estimate must reflect that
    lo = out["spectral_density"][:len(out["freqs"]) // 4].mean()
    hi = out["spectral_density"][-len(out["freqs"]) // 4:].mean()
    assert lo > hi
    with pytest.raises(ValueError):
        ghosal_spec_dens_con(y[:4])


def test_dp_survival_converges_to_kaplan_meier_as_alpha_vanishes():
    rng = np.random.default_rng(9)
    t = rng.exponential(2.0, 120)
    ev = (rng.random(120) > 0.25).astype(float)
    far = ghosal_surv_dp_km(t, ev, alpha=50.0)["max_abs_diff_to_km"]
    near = ghosal_surv_dp_km(t, ev, alpha=0.01)["max_abs_diff_to_km"]
    # the section's whole claim, as a measurement
    assert near < far
    assert near < 1e-2
    out = ghosal_surv_dp_km(t, ev, alpha=1.0)
    assert np.all(np.diff(out["survival_km"]) <= 1e-12)   # non-increasing
    assert out["n_events"] == int(ev.sum())
    with pytest.raises(ValueError):
        ghosal_surv_dp_km(-t, ev)


def test_cox_bvm_recovers_beta_and_reports_efficiency():
    rng = np.random.default_rng(10)
    n = 400
    z = rng.standard_normal(n)
    t = rng.exponential(1.0, n) / np.exp(0.8 * z)
    ev = (rng.random(n) > 0.2).astype(float)
    out = ghosal_cox_bvm(z, time=t, event=ev)
    assert abs(out["beta"][0] - 0.8) < 0.25
    assert out["efficient"] is True
    assert out["credible_equals_confidence"] is True
    assert out["se"][0] > 0
    # the posterior approximation integrates to one
    assert np.trapezoid(out["posterior_normal"], out["beta_grid"]) == \
        pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError):
        ghosal_cox_bvm(z)                      # time is required


def test_empirical_bayes_alpha_tracks_the_cluster_count():
    # the marginal depends on the data only through the partition, so
    # more distinct values must push alpha-hat up
    few = np.repeat(np.arange(3.0), 40)
    many = np.arange(120.0)
    a_few = ghosal_empirical_bayes_np(few)["alpha_hat"]
    a_many = ghosal_empirical_bayes_np(many)["alpha_hat"]
    assert a_many > a_few
    out = ghosal_empirical_bayes_np(few)
    assert out["n_clusters"] == 3
    assert out["understates_uncertainty"] is True
    with pytest.raises(ValueError):
        ghosal_empirical_bayes_np(many, alpha_grid=[-1.0, 1.0])
