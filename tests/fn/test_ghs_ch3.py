"""Known-answer tests for Ghosal Ch 3 ghs008-033."""
import math

from morie.fn.ghs008 import ghosal_ch3_normalized_weights_prior
from morie.fn.ghs009 import ghosal_ch3_stick_breaking_weights
from morie.fn.ghs010 import ghosal_ch3_discrete_hazard_rate
from morie.fn.ghs011 import ghosal_ch3_countable_dirichlet_marginal
from morie.fn.ghs012 import ghosal_ch3_countable_dirichlet_posterior_l
from morie.fn.ghs013 import ghosal_ch3_countable_dirichlet_posterior_k
from morie.fn.ghs014 import ghosal_ch3_dirichlet_posterior_mean
from morie.fn.ghs015 import ghosal_ch3_dirichlet_posterior_var
from morie.fn.ghs016 import ghosal_ch3_dirichlet_posterior_cov
from morie.fn.ghs017 import ghosal_ch3_discrete_random_measure
from morie.fn.ghs018 import ghosal_ch3_tree_splitting_variables
from morie.fn.ghs019 import ghosal_ch3_tree_set_probability
from morie.fn.ghs020 import ghosal_ch3_tree_countable_additivity
from morie.fn.ghs021 import ghosal_ch3_tailfree_max_bound
from morie.fn.ghs023 import ghosal_ch3_tailfree_abs_continuity_cond
from morie.fn.ghs024 import ghosal_ch3_tailfree_canonical_summability
from morie.fn.ghs025 import ghosal_ch3_tailfree_density_product
from morie.fn.ghs026 import ghosal_ch3_tailfree_finite_density_pm
from morie.fn.ghs027 import ghosal_ch3_tailfree_strong_support_event
from morie.fn.ghs028 import ghosal_ch3_polya_tree_first_two_moments
from morie.fn.ghs029 import ghosal_ch3_polya_tree_density_moments
from morie.fn.ghs030 import ghosal_ch3_polya_tree_posterior_density
from morie.fn.ghs031 import ghosal_ch3_polya_tree_mixture_post_density
from morie.fn.ghs032 import ghosal_ch3_polya_tree_density_bounds
from morie.fn.ghs033 import ghosal_ch3_polya_tree_mixture_second_kind


def test_normalized_weights():
    r = ghosal_ch3_normalized_weights_prior([2.0, 3.0, 5.0], k=2)
    assert abs(r["estimate"] - 0.5) < 1e-12
    assert abs(sum(r["distribution"]) - 1.0) < 1e-12


def test_stick_breaking_hand_computed():
    # V = (1/2, 1/3, 1/4): p = (1/2, 1/6, 1/12)
    r = ghosal_ch3_stick_breaking_weights([0.5, 1 / 3, 0.25])
    assert abs(r["distribution"][0] - 0.5) < 1e-12
    assert abs(r["distribution"][1] - 1 / 6) < 1e-12
    assert abs(r["distribution"][2] - 1 / 12) < 1e-12


def test_hazard_inverts_stick_breaking():
    p = ghosal_ch3_stick_breaking_weights([0.3, 0.6, 0.2])["distribution"]
    V = ghosal_ch3_discrete_hazard_rate(p)["value"]
    assert abs(V[0] - 0.3) < 1e-12
    assert abs(V[1] - 0.6) < 1e-12
    assert abs(V[2] - 0.2) < 1e-12


def test_countable_marginal_tail_aggregation():
    r = ghosal_ch3_countable_dirichlet_marginal(
        [1.0, 2.0, 3.0, 4.0], k=2)
    assert r["distribution"] == [1.0, 2.0, 7.0]
    assert abs(sum(r["mean"]) - 1.0) < 1e-12


def test_posterior_l_and_k_updating():
    # alpha = (1,2), counts (3,1), n=4
    rl = ghosal_ch3_countable_dirichlet_posterior_l(
        [1.0, 2.0], [3, 1], l=2, alpha_tail=0.5)
    assert rl["posterior"][:2] == [4.0, 3.0]
    assert abs(rl["posterior"][2] - 0.5) < 1e-12   # 0.5 + 4 - 4
    rk = ghosal_ch3_countable_dirichlet_posterior_k(
        [1.0, 2.0], [3, 1], k=1, alpha_tail=2.0)
    assert rk["posterior"][0] == 4.0
    assert abs(rk["posterior"][1] - 3.0) < 1e-12    # 2 + 4 - 3


def test_posterior_moments_eq37():
    # alpha=(1,2,3), A=6, counts=(2,0,2), n=4: E p_0 = 3/10
    m = ghosal_ch3_dirichlet_posterior_mean(
        [1, 2, 3], [2, 0, 2], 0, 6.0)["value"]
    assert abs(m - 0.3) < 1e-12
    v = ghosal_ch3_dirichlet_posterior_var(
        [1, 2, 3], [2, 0, 2], 0, 6.0)["value"]
    assert abs(v - 0.3 * 0.7 / 11.0) < 1e-12
    c = ghosal_ch3_dirichlet_posterior_cov(
        [1, 2, 3], [2, 0, 2], 0, 1, 6.0)["value"]
    assert abs(c + 0.3 * 0.2 / 11.0) < 1e-12


def test_discrete_measure_mean():
    r = ghosal_ch3_discrete_random_measure([1, 1, 2], [0.0, 1.0, 2.0])
    assert abs(r["estimate"] - (0.25 * 1.0 + 0.5 * 2.0)) < 1e-12


def test_splitting_variables_conditional():
    r = ghosal_ch3_tree_splitting_variables([0.4, 0.1, 0.3])
    assert abs(r["value"][0] - 0.25) < 1e-12
    assert abs(r["value"][1] - 0.75) < 1e-12
    assert r["complement_gap"] < 1e-12


def test_branch_product():
    r = ghosal_ch3_tree_set_probability([0.5, 0.4, 0.25])
    assert abs(r["value"] - 0.05) < 1e-12


def test_atom_mass_vanishes():
    r = ghosal_ch3_tree_countable_additivity(0.5, depth=60)
    assert r["vanishes"] is True
    assert r["value"] == 0.5 ** 60


def test_max_bound_eq314():
    # Beta(a,a) splits: E V^2 = a(a+1)/((2a)(2a+1)); a=1 -> 1/3
    ev2 = [1.0 / 3.0] * 4
    r = ghosal_ch3_tailfree_max_bound(ev2, m=4)
    assert abs(r["value"] - 2.0 ** 4 * (1.0 / 3.0) ** 4) < 1e-12
    assert r["bound_holds"] is True
    # upper: r_j = 2/3, so 2^4 (1/3)^4 == 2^4 (r/2)^4 exactly here
    assert abs(r["upper_bound"] - r["value"]) < 1e-12


def test_abs_continuity_ratio_canonical():
    # a_m = m^2: E V^2 = a(a+1)/((2a)(2a+1)) -> ratio_m =
    # prod 4 E(V^2) stays bounded (abs continuity holds)
    ev2 = [m * m * (m * m + 1.0) / ((2.0 * m * m)
           * (2.0 * m * m + 1.0)) for m in range(1, 25)]
    r = ghosal_ch3_tailfree_abs_continuity_cond(ev2)
    assert r["finite"] is True
    assert r["value"] < 10.0
    # sanity: level-1 ratio = 4 * (2/(2*3)) = 4/3
    assert abs(r["ratios_by_level"][0] - 4.0 / 3.0) < 1e-12


def test_canonical_summability_a_m_squared():
    # symmetric Beta(m^2, m^2): E V = 1/2 exactly; var = 1/(4(2m^2+1))
    ev = [0.5] * 30
    vv = [1.0 / (4.0 * (2.0 * m * m + 1.0)) for m in range(1, 31)]
    r = ghosal_ch3_tailfree_canonical_summability(ev, vv)
    assert r["mean_series"] == 0.0
    assert r["var_series"] < 0.25
    assert r["summable"] is True


def test_density_product_318():
    # V = 1/2 everywhere: p(x) = 1 (uniform)
    r = ghosal_ch3_tailfree_density_product([0.5] * 10)
    assert abs(r["distribution"] - 1.0) < 1e-12
    r2 = ghosal_ch3_tailfree_density_product([0.75, 0.5])
    assert abs(r2["distribution"] - 1.5) < 1e-12


def test_finite_density_pm():
    # depth 2, masses (0.1, 0.2, 0.3, 0.4); x=0.3 -> cell 01 -> 0.2*4
    r = ghosal_ch3_tailfree_finite_density_pm(
        0.3, [0.1, 0.2, 0.3, 0.4], 2)
    assert abs(r["distribution"] - 0.8) < 1e-12
    assert r["cell_index"] == 1


def test_strong_support_product():
    r = ghosal_ch3_tailfree_strong_support_event(0.4, 0.5)
    assert abs(r["value"] - 0.2) < 1e-12
    assert r["positive"] is True


def test_pt_moments_eq321():
    # two levels, Beta(1,1) each: E P = 1/4; E P^2 = (1*2/(2*3))^2 = 1/9
    pairs = [(1.0, 1.0), (1.0, 1.0)]
    r = ghosal_ch3_polya_tree_first_two_moments(pairs)
    assert abs(r["value"][0] - 0.25) < 1e-12
    assert abs(r["value"][1] - 1.0 / 9.0) < 1e-12
    assert abs(r["variance"] - (1.0 / 9.0 - 1.0 / 16.0)) < 1e-12


def test_pt_density_moments_eq322():
    # canonical a_m = m^2, two levels: E p = 1 (uniform);
    # E p^2 = prod 4 a(a+1)/((2a)(2a+1)) = (4*2/(2*3)) * (4*4*5/(8*9))
    pairs = [(1.0, 1.0), (4.0, 4.0)]
    r = ghosal_ch3_polya_tree_density_moments(pairs)
    assert abs(r["value"][0] - 1.0) < 1e-12
    expect_m2 = (4.0 * 1.0 * 2.0 / (2.0 * 3.0)) \
        * (4.0 * 4.0 * 5.0 / (8.0 * 9.0))
    assert abs(r["value"][1] - expect_m2) < 1e-12


def test_pt_posterior_density_eq323():
    # no data: prior mean density is exactly 1
    r0 = ghosal_ch3_polya_tree_posterior_density(0.3, [])
    assert abs(r0["posterior"] - 1.0) < 1e-12
    # data on x's path raise the density there, lower it elsewhere
    hit = ghosal_ch3_polya_tree_posterior_density(0.3, [0.3] * 30)
    miss = ghosal_ch3_polya_tree_posterior_density(0.9, [0.3] * 30)
    assert hit["posterior"] > 1.0 > miss["posterior"]
    # hand check depth 1: a_1=1, N=(30 obs left cell), n=30:
    # factor_1 = (2+60)/(2+30) = 62/32
    assert hit["path_counts"][0] == 30


def test_pt_mixture_posterior():
    # G_theta = identity on [0,1], g_theta = 1: reduces to eq. 3.23
    plain = ghosal_ch3_polya_tree_posterior_density(0.3, [0.3] * 10)
    mix = ghosal_ch3_polya_tree_mixture_post_density(
        0.3, [0.3] * 10, lambda x: 1.0, lambda x: x)
    assert abs(mix["posterior"] - plain["posterior"]) < 1e-12
    # nontrivial g_theta scales the density
    mix2 = ghosal_ch3_polya_tree_mixture_post_density(
        0.3, [0.3] * 10, lambda x: 2.0, lambda x: x)
    assert abs(mix2["posterior"] - 2.0 * plain["posterior"]) < 1e-12


def test_pt_density_bounds_sandwich():
    a = lambda j: float(j * j)
    r = ghosal_ch3_polya_tree_density_bounds(5, a, m=3, depth=40)
    assert r["lower"] <= 1.0 <= r["upper"]
    assert r["lower"] > 0.0
    # deeper m tightens the bracket toward 1
    r2 = ghosal_ch3_polya_tree_density_bounds(5, a, m=10, depth=40)
    assert r2["lower"] > r["lower"]
    assert r2["upper"] < r["upper"]


def test_pt_mixture_second_kind():
    # theta scales alpha symmetrically -> each g_theta = 1, mix = 1
    ap = lambda th, x: [(th, th), (th, th)]
    r = ghosal_ch3_polya_tree_mixture_second_kind(
        0.3, ap, [1.0, 2.0, 5.0])
    assert abs(r["distribution"] - 1.0) < 1e-12
    # asymmetric alphas give a nonuniform mean density
    ap2 = lambda th, x: [(2.0 * th, th)]
    r2 = ghosal_ch3_polya_tree_mixture_second_kind(0.2, ap2, [1.0])
    assert abs(r2["distribution"] - 2.0 * 2.0 / 3.0) < 1e-12
