"""Tests for Ghosal Ch 6-7 consistency modules."""
import math

from morie.fn.gh_c6_1 import ghosal_weak_consist
from morie.fn.gh_c6_2 import ghosal_strong_consist
from morie.fn.gh_c6_3 import ghosal_doob_consist
from morie.fn.gh_c6_4 import ghosal_df_inconsist
from morie.fn.gh_c6_6 import ghosal_kl_support
from morie.fn.gh_c6_7 import ghosal_kl_diverge
from morie.fn.gh_c6_8 import ghosal_tailfree_con
from morie.fn.gh_c6_9 import ghosal_kl_perm
from morie.fn.gh_c6_10 import ghosal_non_iid_con
from morie.fn.gh_c6_11 import ghosal_markov_con
from morie.fn.gh_c6_13 import ghosal_lecam_consist
from morie.fn.gh_c6_14 import ghosal_pred_consist
from morie.fn.gh_c6_16 import ghosal_alpha_post
from morie.fn.gh_c7_1 import ghosal_pt_kl_prop
from morie.fn.gh_c7_2 import ghosal_kern_mix_kl
from morie.fn.gh_c7_3 import ghosal_exp_dens_kl
from morie.fn.gh_c7_5 import ghosal_dpm_gen_con
from morie.fn.gh_c7_8 import ghosal_loc_semipara
from morie.fn.gh_c7_9 import ghosal_linreg_unk_err
from morie.fn.gh_c7_10 import ghosal_mono_reg_con
from morie.fn.gh_iid_consist import ghosal_iid_posterior_consistency
from morie.fn.gh_dp_kl_nbhd import ghosal_dp_kl_nbhd_mass
from morie.fn.gh_ppt_consist import ghosal_polya_tree_consist_rate
from morie.fn.gh_dp_reg_post import ghosal_dp_regression_posterior


def test_weak_consistency_mass_vanishes():
    r = ghosal_weak_consist()
    assert r["decreasing"] is True
    assert r["estimate"] < 1e-3


def test_strong_consistency_path():
    r = ghosal_strong_consist()
    assert r["path_masses"][-1] < r["path_masses"][0]
    assert r["estimate"] < 0.01


def test_doob_posterior_mean_converges():
    r = ghosal_doob_consist()
    assert r["final_error"] < 0.05


def test_df_mechanism_delta_dominates():
    r = ghosal_df_inconsist()
    assert r["delta_component_wins"] is True


def test_kl_support_positive():
    r = ghosal_kl_support([0.25, 0.25, 0.25, 0.25])
    assert r["kl_property"] is True
    assert r["estimate"] > 0.05


def test_kl_divergence_exact():
    # KL((1/2,1/2), (1/4,3/4)) = 0.5 log2 + 0.5 log(2/3)
    r = ghosal_kl_diverge([0.5, 0.5], [0.25, 0.75])
    expect = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    assert abs(r["estimate"] - expect) < 1e-12
    assert ghosal_kl_diverge([0.3, 0.7], [0.3, 0.7])["estimate"] \
        < 1e-12


def test_tailfree_multinomial_consistency():
    r = ghosal_tailfree_con()
    assert r["improving"] is True
    assert r["estimate"] < 0.06


def test_kl_additivity_products():
    r = ghosal_kl_perm([0.5, 0.5], [0.3, 0.7],
                       [0.4, 0.6], [0.35, 0.65])
    assert r["additivity_gap"] < 1e-12
    assert abs(r["estimate"] - sum(r["kl_marginals"])) < 1e-12


def test_non_iid_posterior_concentrates():
    r = ghosal_non_iid_con()
    assert r["error_by_n"][-1] < 0.35
    assert r["avg_kl_at_delta_0.1"] < 0.01


def test_markov_transition_kl_small():
    r = ghosal_markov_con()
    assert r["estimate"] < 0.01
    assert abs(r["a_hat"] - 0.3) < 0.06
    assert abs(r["b_hat"] - 0.6) < 0.06


def test_lecam_bound_arithmetic():
    r = ghosal_lecam_consist(0.1, 0.05, 0.5, 0.02)
    assert abs(r["estimate"] - (0.1 + 0.05 + 0.04)) < 1e-12


def test_predictive_cesaro_kl_decays():
    r = ghosal_pred_consist()
    assert r["decaying"] is True
    assert r["estimate"] < 0.02


def test_alpha_posterior_exact_and_wider():
    r = ghosal_alpha_post(30, 100, alpha=0.5)
    assert r["alpha_posterior"] == [16.0, 36.0]
    assert r["wider_than_full"] is True


def test_pt_kl_series():
    ok = ghosal_pt_kl_prop(2.0)
    assert ok["kl_property"] is True
    assert ok["estimate"] < 2.0            # sum 1/m^2 < pi^2/6
    bad = ghosal_pt_kl_prop(1.0)
    assert bad["kl_property"] is False


def test_kernel_mixture_kl_improves():
    r = ghosal_kern_mix_kl()
    assert r["improving"] is True
    assert r["estimate"] < 0.01


def test_exp_link_kl_controlled():
    r = ghosal_exp_dens_kl()
    assert r["kl_small_when_sup_small"] is True
    same = ghosal_exp_dens_kl(coefs=(0.5, -0.3))
    assert same["estimate"] < 1e-10


def test_dpm_consistency_error_falls():
    r = ghosal_dpm_gen_con()
    assert r["improving"] is True


def test_semiparametric_location():
    r = ghosal_loc_semipara()
    assert r["error"] < 0.15


def test_linreg_unknown_error():
    r = ghosal_linreg_unk_err()
    assert r["error"] < 0.2


def test_monotone_regression():
    r = ghosal_mono_reg_con()
    assert r["monotone"] is True
    assert r["estimate"] < 0.15


def test_schwartz_exponential_decay():
    r = ghosal_iid_posterior_consistency()
    assert r["exponential"] is True
    assert r["estimate"] < 1e-4


def test_kl_nbhd_mass_positive_and_monotone():
    r = ghosal_dp_kl_nbhd_mass([0.25] * 4)
    assert r["monotone"] is True
    assert r["log_mass_by_eps"][0] > math.log(1e-3)


def test_pt_contraction():
    r = ghosal_polya_tree_consist_rate()
    assert r["contracting"] is True


def test_dp_error_regression():
    r = ghosal_dp_regression_posterior()
    assert r["error"] < 0.2
