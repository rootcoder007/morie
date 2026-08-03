"""Tests for Ghosal Ch 4 tail + Ch 5 DPM modules."""
import math

from morie.fn.gh_c4_16 import ghosal_dp_tails
from morie.fn.gh_c4_17 import ghosal_dp_median
from morie.fn.gh_c4_18 import ghosal_dp_mean_dist
from morie.fn.gh_c4_19 import ghosal_dp_charact
from morie.fn.gh_c4_20 import ghosal_mix_dp
from morie.fn.gh_c4_21 import ghosal_inv_dp
from morie.fn.gh_c4_22 import ghosal_constr_dp
from morie.fn.gh_c4_23 import ghosal_pen_dp
from morie.fn.gh_c4_24 import ghosal_bayes_boot
from morie.fn.gh_c5_1 import ghosal_dpm_model
from morie.fn.gh_c5_2 import ghosal_dpm_marg
from morie.fn.gh_c5_3 import ghosal_cgibbs
from morie.fn.gh_c5_4 import ghosal_splitmerge
from morie.fn.gh_c5_5 import ghosal_blk_gibbs
from morie.fn.gh_c5_6 import ghosal_vb_dpm
from morie.fn.gh_c5_10 import ghosal_poi_ker

TWO_CLUSTERS = [-2.1, -1.9, -2.0, -2.2, 1.9, 2.0, 2.1, 1.8]


def test_dp_tails_thinner_than_base():
    r = ghosal_dp_tails(0.01, r=2.0)
    assert r["thinner_than_base"] is True
    assert r["lower"] <= r["upper"]


def test_dp_median_symmetry_and_mass():
    # G(x)=1/2: Beta(M/2, M/2) symmetric about 1/2 -> H = 1/2
    r = ghosal_dp_median(0.5, 4.0)
    assert abs(r["estimate"] - 0.5) < 1e-3
    # G(x) large: F(x) almost surely > 1/2 -> H near 1
    hi = ghosal_dp_median(0.95, 20.0)
    assert hi["estimate"] > 0.95


def test_dp_mean_dist_two_point_base():
    # alpha = theta0*delta_0 + theta1*delta_1: mean ~ Be(theta1,
    # theta0) (Problem 4.29). H(0.5) for Be(1,1) = 0.5.
    r = ghosal_dp_mean_dist([0.0, 1.0], [1.0, 1.0], 0.5)
    assert abs(r["estimate"] - 0.5) < 0.02
    # Be(2,1): H(0.5) = 0.25
    r2 = ghosal_dp_mean_dist([0.0, 1.0], [1.0, 2.0], 0.5)
    assert abs(r2["estimate"] - 0.25) < 0.02


def test_neutrality_of_dirichlet():
    r = ghosal_dp_charact([2.0, 3.0, 4.0], seed=11)
    assert r["neutral"] is True


def test_mdp_moments():
    # two components equal weight, G0 = 0.2 / 0.6, M = 4 both
    r = ghosal_mix_dp([0.2, 0.6], [4.0, 4.0], [0.5, 0.5])
    assert abs(r["estimate"] - 0.4) < 1e-12
    w_in = 0.5 * (0.2 * 0.8 / 5.0) + 0.5 * (0.6 * 0.4 / 5.0)
    assert abs(r["var_within"] - w_in) < 1e-12
    assert abs(r["var_between"] - 0.04) < 1e-12


def test_invariant_dp_symmetric_estimator():
    # symmetric data, x=0: count = n/2 exactly when no ties at 0
    r = ghosal_inv_dp(0.0, [-1.0, -0.5, 0.5, 1.0], 0.5, 1.0)
    assert abs(r["symmetrized_count"] - 2.0) < 1e-12
    assert abs(r["estimate"] - 2.5 / 5.0) < 1e-12


def test_constrained_dp_respects_controls():
    r = ghosal_constr_dp([0.3, 0.7], [[1.0, 1.0], [2.0, 2.0]])
    assert abs(sum(r["P_cells"][:2]) - 0.3) < 1e-12
    assert abs(r["total_mass"] - 1.0) < 1e-12


def test_penalized_dirichlet_prefers_smooth():
    smooth = ghosal_pen_dp([0.25, 0.25, 0.25, 0.25],
                           [1.0] * 4, lam=10.0)
    rough = ghosal_pen_dp([0.7, 0.1, 0.1, 0.1], [1.0] * 4, lam=10.0)
    assert smooth["estimate"] > rough["estimate"]
    post = ghosal_pen_dp([0.25] * 4, [1.0] * 4, 1.0,
                         counts=[3, 1, 0, 0])
    assert post["posterior_alpha"] == [4.0, 2.0, 1.0, 1.0]


def test_bayesian_bootstrap_centres_on_sample_mean():
    data = [1.0, 2.0, 3.0, 4.0]
    r = ghosal_bayes_boot(data, n_draws=400, seed=7)
    assert abs(r["estimate"] - 2.5) < 0.15


def test_dpm_density_positive_and_finite():
    r = ghosal_dpm_model([0.5, 5.0])
    assert r["density"][0] > r["density"][1]
    assert r["mixing_mass"] > 0.99


def test_dpm_marginal_prefers_clustered_data():
    tight = ghosal_dpm_marg([0.0, 0.01, -0.01, 0.02])
    spread = ghosal_dpm_marg([0.0, 3.0, -3.0, 6.0])
    assert tight["estimate"] > spread["estimate"]


def test_cgibbs_finds_two_clusters():
    r = ghosal_cgibbs(TWO_CLUSTERS, seed=13)
    assert 2 <= r["n_clusters"] <= 3


def test_splitmerge_accepts_good_split():
    z = [0] * 8
    r = ghosal_splitmerge(TWO_CLUSTERS, z, split_label=0)
    assert r["estimate"] == 1.0            # split clearly improves
    assert len(set(r["z_proposed"])) == 2


def test_blocked_gibbs_two_clusters():
    r = ghosal_blk_gibbs(TWO_CLUSTERS, seed=17)
    assert 2 <= r["n_active"] <= 3
    assert abs(sum(r["weights"]) - 1.0) < 1e-9


def test_vb_centers_near_truth():
    r = ghosal_vb_dpm(TWO_CLUSTERS, seed=19)
    cs = sorted(r["centers"])
    assert min(abs(c + 2.0) for c in cs) < 0.3
    assert min(abs(c - 2.0) for c in cs) < 0.3


def test_poisson_mixture_known_weights():
    # single atom lambda=2: pmf(2) = e^-2 2^2/2!
    r = ghosal_poi_ker([2], lambdas=[2.0], weights=[1.0])
    assert abs(r["estimate"] - math.exp(-2.0) * 2.0) < 1e-12
    # two atoms mix linearly
    r2 = ghosal_poi_ker([0], lambdas=[1.0, 2.0], weights=[1, 1])
    expect = 0.5 * math.exp(-1.0) + 0.5 * math.exp(-2.0)
    assert abs(r2["estimate"] - expect) < 1e-12
