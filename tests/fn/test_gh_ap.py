"""Tests for Ghosal appendix modules."""
import math

from morie.fn.gh_ap_a1 import ghosal_weak_conv_def
from morie.fn.gh_ap_a2 import ghosal_prohorov_metric
from morie.fn.gh_ap_a3 import ghosal_tv_distance
from morie.fn.gh_ap_a4 import ghosal_hellinger_dist
from morie.fn.gh_ap_b1 import ghosal_kl_props
from morie.fn.gh_ap_b2 import ghosal_kl_variation
from morie.fn.gh_ap_b3 import ghosal_renyi_div
from morie.fn.gh_ap_c1 import ghosal_covering_num
from morie.fn.gh_ap_c2 import ghosal_packing_num
from morie.fn.gh_ap_c3 import ghosal_bracket_num
from morie.fn.gh_ap_e1 import ghosal_bernstein_poly
from morie.fn.gh_ap_e2 import ghosal_spline_space
from morie.fn.gh_ap_e3 import ghosal_wavelet_mra
from morie.fn.gh_ap_f1 import ghosal_donsker_class
from morie.fn.gh_ap_f2 import ghosal_glivenko
from morie.fn.gh_ap_g1 import ghosal_fin_dir_def
from morie.fn.gh_ap_g2 import ghosal_dir_moments
from morie.fn.gh_ap_g3 import ghosal_dir_marginal
from morie.fn.gh_ap_h1 import ghosal_inv_gauss
from morie.fn.gh_ap_i1 import ghosal_gp_sample_cont
from morie.fn.gh_ap_i2 import ghosal_dudley_entropy
from morie.fn.gh_ap_i3 import ghosal_borell_tis
from morie.fn.gh_ap_j1 import ghosal_levy_ito
from morie.fn.gh_ap_j2 import ghosal_crm_laplace
from morie.fn.gh_crm_def import ghosal_completely_random_measure
from morie.fn.gh_ap_k2 import ghosal_assouad_lemma
from morie.fn.gh_ap_m1 import ghosal_mh_sampler
from morie.fn.gh_ap_m2 import ghosal_gibbs_sampler
from morie.fn.gh_ap_m3 import ghosal_slice_sampler

P = [0.5, 0.5]
Q = [0.25, 0.75]


def test_weak_convergence():
    assert ghosal_weak_conv_def()["converging"] is True


def test_prohorov_bounded_by_tv():
    r = ghosal_prohorov_metric(P, Q)
    assert abs(r["estimate"] - 0.25) < 1e-12


def test_tv_forms_agree():
    r = ghosal_tv_distance(P, Q)
    assert r["forms_agree"] is True
    assert abs(r["estimate"] - 0.25) < 1e-12


def test_hellinger_inequalities():
    r = ghosal_hellinger_dist(P, Q)
    assert r["inequalities_hold"] is True
    expected = 1.0 - (math.sqrt(0.5 * 0.25) + math.sqrt(0.5 * 0.75))
    assert abs(r["estimate"] - expected) < 1e-12


def test_kl_pinsker():
    r = ghosal_kl_props(P, Q)
    assert r["nonneg"] and r["pinsker_holds"]
    assert r["zero_iff_equal"] is True
    same = ghosal_kl_props(P, P)
    assert same["estimate"] < 1e-14


def test_kl_variation_v1_vs_kl():
    r = ghosal_kl_variation(P, Q, k=1)
    assert r["estimate"] >= abs(r["kl"]) - 1e-12


def test_renyi_half_hellinger_link():
    r = ghosal_renyi_div(P, Q, alpha=0.5)
    assert r["hellinger_link_gap"] < 1e-12


def test_covering_packing_bracketing():
    c = ghosal_covering_num()
    assert c["lower"] <= c["upper"]
    p = ghosal_packing_num()
    assert p["relation_holds"] is True
    b = ghosal_bracket_num(1.0, 0.1)
    assert abs(b["estimate"] - 10.0) < 1e-12


def test_bernstein_error_shrinks():
    assert ghosal_bernstein_poly()["improving"] is True


def test_spline_dim():
    r = ghosal_spline_space()
    assert r["estimate"] == 14.0
    assert abs(r["approx_error_order"] - 0.01) < 1e-12


def test_haar_parseval_converges():
    r = ghosal_wavelet_mra(10)
    assert r["parseval_gap"] < 1e-4


def test_donsker_threshold():
    assert ghosal_donsker_class(1.0)["donsker"] is True
    assert ghosal_donsker_class(0.4)["donsker"] is False


def test_glivenko_cantelli():
    assert ghosal_glivenko()["vanishing"] is True


def test_dirichlet_density_and_moments():
    g1 = ghosal_fin_dir_def()
    assert g1["estimate"] > 0
    g2 = ghosal_dir_moments([2.0, 3.0, 5.0])
    assert abs(g2["estimate"] - 0.2) < 1e-12
    assert abs(g2["variance"] - 2.0 * 8.0 / (100.0 * 11.0)) < 1e-12
    assert abs(g2["covariance"] + 6.0 / (100.0 * 11.0)) < 1e-12


def test_dirichlet_aggregation():
    r = ghosal_dir_marginal([2.0, 3.0, 5.0], (0, 1))
    assert r["beta_params"] == [5.0, 5.0]
    assert abs(r["estimate"] - 0.5) < 1e-12


def test_inverse_gaussian_normalized():
    assert ghosal_inv_gauss()["normalized"] is True


def test_kolmogorov_continuity():
    r = ghosal_gp_sample_cont()
    assert abs(r["estimate"] - 0.5) < 1e-12


def test_dudley_finiteness():
    fin = ghosal_dudley_entropy(entropy_exponent=1.0)
    assert fin["finite"] is True
    inf = ghosal_dudley_entropy(entropy_exponent=2.5)
    assert inf["finite"] is False


def test_borell_tis_monotone():
    assert ghosal_borell_tis()["tighter_for_larger_u"] is True


def test_levy_ito_pieces():
    r = ghosal_levy_ito()
    assert abs(r["estimate"] - r["fixed_mass"]
               - r["poisson_mass"]) < 1e-12


def test_crm_laplace_closed_form():
    assert ghosal_crm_laplace()["gap"] < 5e-3


def test_crm_independence():
    assert ghosal_completely_random_measure()["independent"] is True


def test_assouad_value():
    r = ghosal_assouad_lemma()
    assert abs(r["estimate"] - 0.32) < 1e-12


def test_mh_targets_standard_normal():
    r = ghosal_mh_sampler()
    assert abs(r["mean"]) < 0.15
    assert abs(r["estimate"] - 1.0) < 0.25
    assert 0.2 < r["accept_rate"] < 0.9


def test_gibbs_recovers_correlation():
    assert ghosal_gibbs_sampler()["gap"] < 0.08


def test_slice_sampler_exp_mean():
    assert ghosal_slice_sampler()["gap"] < 0.15
