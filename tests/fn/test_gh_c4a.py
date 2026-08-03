"""Tests for Ghosal Ch 4 DP core (gh_c4_1..15 + cond dist)."""
import math

from morie.fn.gh_c4_1 import ghosal_dp_def
from morie.fn.gh_c4_2 import ghosal_dp_mean
from morie.fn.gh_c4_3 import ghosal_dp_var
from morie.fn.gh_c4_4 import ghosal_dp_cov
from morie.fn.gh_c4_5 import ghosal_dp_selfsim
from morie.fn.gh_dp_cond_dist import ghosal_dp_conditional_distribution
from morie.fn.gh_c4_6 import ghosal_dp_post
from morie.fn.gh_c4_7 import ghosal_dp_pred
from morie.fn.gh_c4_8 import ghosal_dp_ndist
from morie.fn.gh_c4_9 import ghosal_dp_gamma
from morie.fn.gh_c4_10 import ghosal_dp_polya_urn
from morie.fn.gh_c4_11 import ghosal_dp_stickbr
from morie.fn.gh_c4_12 import ghosal_dp_discrete
from morie.fn.gh_c4_13 import ghosal_dp_weak_conv
from morie.fn.gh_c4_14 import ghosal_dp_fs_approx
from morie.fn.gh_c4_15 import ghosal_dp_mutual_sing


def test_dp_def_simplex():
    r = ghosal_dp_def([1.0, 2.0, 3.0])
    assert abs(sum(r["P"]) - 1.0) < 1e-12
    assert min(r["P"]) > 0


def test_dp_moments_hand():
    assert ghosal_dp_mean(0.3)["estimate"] == 0.3
    # var: 0.3*0.7/(1+4) = 0.042
    assert abs(ghosal_dp_var(0.3, 4.0)["estimate"] - 0.042) < 1e-12
    # disjoint A,B: G0(AB)=0 -> cov = -0.3*0.2/5
    assert abs(ghosal_dp_cov(0.0, 0.3, 0.2, 4.0)["estimate"]
               + 0.012) < 1e-12
    # A subset B: G0(AB)=G0(A)
    assert ghosal_dp_cov(0.3, 0.3, 0.5, 4.0)["estimate"] > 0


def test_selfsim_recombines_to_one():
    r = ghosal_dp_selfsim(0.3, [1.0, 1.0], [2.0, 2.0])
    assert abs(r["total_mass"] - 1.0) < 1e-12
    assert abs(sum(r["P_cells"][:2]) - 0.3) < 1e-12


def test_cond_dist_normalized():
    r = ghosal_dp_conditional_distribution([1.0, 3.0])
    assert abs(sum(r["P_cond"]) - 1.0) < 1e-12
    assert r["independent_of_w"] is True


def test_posterior_mean_convex_combination():
    # M=2, n=8, G0(A)=0.5, 6 of 8 in A: 0.2*0.5 + 0.8*0.75 = 0.7
    r = ghosal_dp_post(0.5, 2.0, 6, 8)
    assert abs(r["estimate"] - 0.7) < 1e-12
    assert r["posterior_var"] <= r["var_bound"] + 1e-15


def test_predictive_weights():
    r = ghosal_dp_pred([1.0, 2.0, 2.0], 2.0)
    assert abs(r["weight_fresh"] - 0.4) < 1e-12
    assert abs(r["weight_per_obs"] - 0.2) < 1e-12
    r2 = ghosal_dp_pred([1.0, 2.0, 2.0], 2.0, x_new_equals=2.0)
    assert abs(r2["estimate"] - 0.4) < 1e-12


def test_ndist_exact_and_bounds():
    # M=1: E K_n = H_n
    r = ghosal_dp_ndist(4, 1.0)
    H4 = 1.0 + 0.5 + 1.0 / 3.0 + 0.25
    assert abs(r["estimate"] - H4) < 1e-12
    assert r["bounds_hold"] is True
    # var: sum (i-1)/i^2 for i=1..4
    v = sum((i - 1.0) / i ** 2 for i in range(1, 5))
    assert abs(r["variance"] - v) < 1e-12


def test_gamma_construction_matches_def():
    a = [1.0, 2.0, 3.0]
    assert ghosal_dp_gamma(a, seed=9)["P"] == \
        ghosal_dp_def(a, seed=9)["P"]


def test_polya_urn_ties():
    r = ghosal_dp_polya_urn(200, 2.0, seed=3)
    assert r["n_distinct"] < 200          # ties must occur
    assert r["n_distinct"] >= 1


def test_sethuraman_mass_and_mean():
    # M = 50: var of the DP mean is 1/(12(M+1)) so sd ~ 0.04
    r = ghosal_dp_stickbr(3000, 50.0, seed=4)
    assert r["total_mass"] > 0.999
    assert 0.35 < r["estimate"] < 0.65    # uniform center measure


def test_discreteness_largest_atom():
    r = ghosal_dp_discrete(500, 1.0, seed=5)
    assert r["largest_atom"] > 0.05
    assert r["atoms_carry_all_mass"] is True


def test_weak_conv_regimes():
    lo = ghosal_dp_weak_conv(0.5, [1.0, 0.1, 1e-9])
    assert "random point" in lo["regime"]
    assert abs(lo["estimate"] - 0.25) < 1e-6
    hi = ghosal_dp_weak_conv(0.5, [1.0, 100.0, 1e9])
    assert "center measure" in hi["regime"]
    assert hi["estimate"] < 1e-6


def test_eps_dp_truncation():
    r = ghosal_dp_fs_approx(0.01, 2.0, seed=6)
    assert r["remainder_mass"] <= 0.01
    assert r["tv_bound"] == 0.01
    assert abs(r["expected_support_size"]
               - (2.0 + 2.0 * math.log(100.0))) < 1e-12


def test_mutual_singularity_cases():
    same = ghosal_dp_mutual_sing([0.5, 0.5], [0.5, 0.5],
                                 [1.0], [1.0])
    assert same["mutually_singular"] is False
    diff = ghosal_dp_mutual_sing([0.5, 0.5], [0.4, 0.6],
                                 [1.0], [1.0])
    assert diff["mutually_singular"] is True
    atom = ghosal_dp_mutual_sing([0.5, 0.5], [0.5, 0.5],
                                 [1.0], [2.0])
    assert atom["mutually_singular"] is True
