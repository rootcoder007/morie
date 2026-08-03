"""Tests for Ghosal Ch 11 GP-prior modules."""
import math

from morie.fn.gh_c11_1 import ghosal_gp_def_rkhs
from morie.fn.gh_c11_2 import ghosal_rkhs_norm
from morie.fn.gh_c11_3 import ghosal_gp_crt_thm
from morie.fn.gh_conc_func import ghosal_concentration_function
from morie.fn.gh_small_ball import ghosal_small_ball_prob
from morie.fn.gh_c11_5 import ghosal_gp_binreg_crt
from morie.fn.gh_c11_6 import ghosal_bm_prior
from morie.fn.gh_gp_brow_prim import ghosal_gp_brownian_primitive
from morie.fn.gh_c11_7 import ghosal_rl_process
from morie.fn.gh_c11_8 import ghosal_fbm_prior
from morie.fn.gh_c11_9 import ghosal_statgp_spec
from morie.fn.gh_gp_orn_uhl import ghosal_gp_ornstein_uhlenbeck
from morie.fn.gh_c11_10 import ghosal_series_gp
from morie.fn.gh_c11_11 import ghosal_rescal_gp
from morie.fn.gh_c11_12 import ghosal_selfsim_gp
from morie.fn.gh_c11_13 import ghosal_gp_adapt_thm
from morie.fn.gh_c11_14 import ghosal_gp_laplace
from morie.fn.gh_c11_15 import ghosal_ep_gp
from morie.fn.gh_sup_norm_gp import ghosal_sup_norm_contraction


def test_rkhs_inner_product_and_reproducing():
    S = [[2.0, 0.5], [0.5, 1.0]]
    r = ghosal_gp_def_rkhs(S, [1.0, 0.0], [0.0, 1.0])
    assert abs(r["estimate"] - 0.5) < 1e-12     # a' S b = S_01
    assert r["reproducing_gap"] < 1e-12


def test_concentration_terms_positive():
    r = ghosal_rkhs_norm([1.0, 0.5, 0.25], [1.0, 0.25, 0.0625], 0.3)
    assert r["estimate"] > 0
    assert r["decentering_norm2"] >= 0


def test_gp_rate_equation_bm():
    r = ghosal_gp_crt_thm(2.0, 10000)
    assert abs(r["estimate"] - 0.1) < 1e-12     # 10000^{-1/4}
    assert r["balance_gap"] < 1e-9


def test_concentration_function_sum():
    r = ghosal_concentration_function(1.5, 2.5)
    assert abs(r["estimate"] - 4.0) < 1e-12


def test_small_ball_exponent_grows():
    r = ghosal_small_ball_prob()
    assert r["increasing"] is True
    assert r["estimate"] > r["phi_by_eps"][0]


def test_binreg_rate_values():
    r = ghosal_gp_binreg_crt(2.0, 1.0, ns=(10000,))
    assert abs(r["estimate"] - 10000 ** (-0.4)) < 1e-15


def test_bm_covariance_min():
    r = ghosal_bm_prior()
    assert r["cov_gap"] < 0.05
    assert r["var_gap"] < 0.05


def test_integrated_bm_smoother():
    r = ghosal_gp_brownian_primitive()
    assert r["smoother"] is True


def test_rl_variance_growth():
    r = ghosal_rl_process()
    assert r["gap"] < 0.3


def test_fbm_kernel_properties():
    r = ghosal_fbm_prior()
    assert r["var_gap"] < 1e-12
    assert r["positive_definite"] is True


def test_bochner_square_exponential():
    r = ghosal_statgp_spec()
    assert r["bochner_gap"] < 1e-6


def test_ou_representation_and_markov():
    r = ghosal_gp_ornstein_uhlenbeck()
    assert r["representation_gap"] < 1e-12
    assert r["markov_gap"] < 1e-12


def test_series_gp_converges():
    r = ghosal_series_gp()
    assert r["converging"] is True


def test_rescaling_roughens():
    r = ghosal_rescal_gp()
    assert r["roughens_as_l_shrinks"] is True


def test_fbm_self_similarity_exact():
    r = ghosal_selfsim_gp(0.6, 3.0)
    assert r["gap"] < 1e-12


def test_length_scale_adaptation():
    r = ghosal_gp_adapt_thm()
    assert r["estimate"] == 0.2


def test_laplace_separates_classes():
    r = ghosal_gp_laplace()
    assert r["separates"] is True
    assert r["laplace_var_site0"] > 0


def test_ep_separates_classes():
    r = ghosal_ep_gp()
    assert r["separates"] is True
    assert r["ep_var_site0"] > 0


def test_sup_norm_rate_decreasing():
    r = ghosal_sup_norm_contraction()
    assert r["decreasing"] is True
