"""Pitman-Yor, dependent DP, Polya urn, difference boundaries, Polya trees."""
import importlib
import math

import pytest

pmpfit = importlib.import_module("morie.fn.pmpfit")
ddpest = importlib.import_module("morie.fn.ddpest")
posspr = importlib.import_module("morie.fn.posspr")
dpgrf = importlib.import_module("morie.fn.dpgrf")
poltrx = importlib.import_module("morie.fn.poltrx")

W_CHAIN = [[0.0, 1.0, 0.0, 0.0],
           [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0],
           [0.0, 0.0, 1.0, 0.0]]
DRAWS = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]]


# ------------------------------------------------------------- pmpfit
def test_dirichlet_process_is_the_alpha_zero_boundary():
    assert pmpfit.check_parameters(0.0, 1.0)["is_dirichlet"]
    assert not pmpfit.check_parameters(0.25, 1.0)["is_dirichlet"]


@pytest.mark.parametrize("alpha,theta", [(1.0, 1.0), (-0.1, 1.0),
                                         (0.5, -0.5), (0.5, -0.6)])
def test_definition_one_range_is_enforced(alpha, theta):
    with pytest.raises(ValueError):
        pmpfit.check_parameters(alpha, theta)


def test_predictive_weights_are_a_probability_vector():
    w = pmpfit.predictive_weights([5.0, 3.0, 1.0], 0.5, 1.0)
    assert w["total"] == pytest.approx(1.0, abs=1e-12)
    assert w["occupied"][0] == pytest.approx(4.5 / 10.0)
    assert w["new"] == pytest.approx(2.5 / 10.0)


def test_the_discount_taken_from_clusters_funds_the_new_one():
    w = pmpfit.predictive_weights([5.0, 3.0, 1.0], 0.5, 1.0)
    assert w["discount_transferred"] == pytest.approx(1.5 / 10.0)
    assert w["new"] == pytest.approx(1.0 / 10.0
                                     + w["discount_transferred"])


def test_alpha_zero_reduces_to_the_plain_polya_urn():
    w = pmpfit.predictive_weights([5.0, 3.0, 1.0], 0.0, 1.0)
    assert w["occupied"] == pytest.approx([0.5, 0.3, 0.1])
    assert w["new"] == pytest.approx(0.1)


def test_expected_clusters_matches_the_digamma_closed_form():
    k = importlib.import_module("morie.fn._s03core")
    got = pmpfit.expected_clusters(1000, 0.0, 1.0)["expected"]
    assert got == pytest.approx(k.digamma(1001.0) - k.digamma(1.0),
                                abs=1e-9)


def test_a_positive_discount_gives_many_more_clusters():
    dp = pmpfit.expected_clusters(1000, 0.0, 1.0)["expected"]
    py = pmpfit.expected_clusters(1000, 0.5, 1.0)["expected"]
    assert py > 3.0 * dp


def test_cluster_count_is_monotone_in_the_discount():
    tc = pmpfit.tail_comparison(500, 1.0, (0.0, 0.3, 0.6))
    assert tc["monotone_in_alpha"]


def test_the_discount_leaves_mass_in_the_stick_breaking_tail():
    s0 = pmpfit.stick_breaking_py(0.0, 1.0, 400, seed=5)
    s6 = pmpfit.stick_breaking_py(0.6, 1.0, 400, seed=5)
    assert sum(s0["weights"]) + s0["remaining"] == pytest.approx(1.0)
    assert s6["remaining"] > s0["remaining"]


def test_a_cluster_smaller_than_the_discount_is_refused():
    with pytest.raises(ValueError):
        pmpfit.predictive_weights([0.3], 0.5, 1.0)


# ------------------------------------------------------------- ddpest
def test_the_two_constructions_vary_different_ingredients():
    assert (ddpest.dependence_kind("single_weights")["varies_with_x"]
            == "atoms")
    assert (ddpest.dependence_kind("single_atoms")["varies_with_x"]
            == "weights")


def test_unknown_construction_is_refused():
    with pytest.raises(ValueError):
        ddpest.dependence_kind("single_everything")


def test_single_weights_shares_the_stick_breaking_draw():
    xs = [0.0, 1.0, 2.0]
    r = ddpest.single_weights_ddp(xs, 1.0, 12, lambda x, h: (h, x),
                                  seed=3)
    assert all(r["G"][x]["weights"] == r["G"][0.0]["weights"]
               for x in xs)
    assert r["G"][0.0]["atoms"] != r["G"][1.0]["atoms"]


def test_single_atoms_shares_the_support():
    xs = [0.0, 1.0, 2.0]
    r = ddpest.single_atoms_ddp(xs, 1.0, 5,
                                lambda x, h: math.exp(-abs(h - x)),
                                seed=3)
    assert r["G"][0.0]["atoms"] == r["G"][2.0]["atoms"]
    assert r["G"][0.0]["weights"] != r["G"][2.0]["weights"]


def test_every_marginal_remains_a_probability_measure():
    xs = [0.0, 1.0]
    a = ddpest.single_weights_ddp(xs, 1.0, 8, lambda x, h: (h, x),
                                  seed=1)
    b = ddpest.single_atoms_ddp(xs, 1.0, 8,
                                lambda x, h: 1.0 / (1.0 + h + x),
                                seed=1)
    assert ddpest.check_marginals(a["G"])["ok"]
    assert ddpest.check_marginals(b["G"])["ok"]


def test_dependence_decays_with_covariate_distance():
    xs = [0.0, 1.0, 2.0]
    r = ddpest.single_atoms_ddp(xs, 1.0, 5,
                                lambda x, h: math.exp(-abs(h - x)),
                                seed=3)
    near = ddpest.correlation(r["G"], 0.0, 1.0, lambda a: a < 2)
    far = ddpest.correlation(r["G"], 0.0, 2.0, lambda a: a < 2)
    assert near["abs_difference"] < far["abs_difference"]
    assert ddpest.correlation(r["G"], 1.0, 1.0,
                              lambda a: a < 2)["identical"]


def test_the_predicted_density_mixes_the_atoms_at_that_covariate():
    r = ddpest.single_atoms_ddp([0.0], 1.0, 5,
                                lambda x, h: 1.0 / (1.0 + h),
                                seed=2)
    d = ddpest.predict_density(r["G"], 0.0, [0.0, 1.0],
                               lambda y, th: math.exp(-(y - th) ** 2))
    assert d["n_components"] == 5
    assert all(v > 0.0 for v in d["density"])


def test_a_negative_weight_is_refused():
    with pytest.raises(ValueError):
        ddpest.single_atoms_ddp([0.0], 1.0, 3, lambda x, h: -1.0)


# ------------------------------------------------------------- posspr
def test_urn_weights_are_a_probability_vector():
    w = posspr.urn_weights([3.0, 1.0], 2.0)
    assert w["total"] == pytest.approx(1.0, abs=1e-12)
    assert w["new"] == pytest.approx(2.0 / 6.0)
    assert w["existing"] == pytest.approx([0.5, 1.0 / 6.0])


@pytest.mark.parametrize("alpha,tie", [(1.0, 0.5), (3.0, 0.25),
                                       (0.5, 2.0 / 3.0)])
def test_tie_probability_is_one_over_one_plus_alpha(alpha, tie):
    assert posspr.tie_probability(alpha)["tie"] == pytest.approx(tie)


def test_simulation_reproduces_the_tie_probability():
    ties = sum(1 for s in range(3000)
               if posspr.sample_urn(2, 1.0, seed=s)["n_clusters"] == 1)
    assert abs(ties / 3000.0 - 0.5) < 0.03


def test_expected_clusters_is_the_sum_of_alpha_over_alpha_plus_i():
    got = posspr.expected_clusters(50, 2.0)["expected"]
    assert got == pytest.approx(sum(2.0 / (2.0 + i)
                                    for i in range(50)))


def test_the_predictive_density_is_computed_by_hand():
    d = posspr.predictive_density([0.0, 1.0], [0.0, 5.0], [3.0, 1.0],
                                  2.0,
                                  lambda y, th: math.exp(-(y - th) ** 2),
                                  lambda y: 0.1)
    assert d["new_cluster_weight"] == pytest.approx(1.0 / 3.0)
    assert d["density"][0] == pytest.approx(
        (1.0 / 3.0) * 0.1 + 0.5 + math.exp(-25.0) / 6.0)


def test_a_non_positive_concentration_is_refused():
    with pytest.raises(ValueError):
        posspr.urn_weights([1.0], 0.0)


def test_sampling_more_customers_never_loses_clusters():
    r = posspr.sample_urn(200, 1.0, seed=7)
    assert 1 <= r["n_clusters"] <= 200
    assert sum(r["counts"]) == 200


# -------------------------------------------------------------- dpgrf
def test_only_adjacent_pairs_are_candidate_boundaries():
    ap = dpgrf.adjacency_pairs(W_CHAIN)
    assert ap["pairs"] == [(0, 1), (1, 2), (2, 3)]
    assert ap["n_pairs"] == 3


def test_an_asymmetric_adjacency_is_refused():
    with pytest.raises(ValueError):
        dpgrf.adjacency_pairs([[0.0, 1.0], [0.0, 0.0]])


def test_a_continuous_prior_puts_zero_mass_on_ties():
    assert (dpgrf.continuous_prior_tie_probability()["probability"]
            == 0.0)


def test_the_intrinsic_car_is_singular_and_reports_it():
    assert dpgrf.car_precision(W_CHAIN, rho=1.0)["singular"]
    assert not dpgrf.car_precision(W_CHAIN, rho=0.9)["singular"]


def test_the_car_precision_is_tau_times_d_minus_rho_w():
    q = dpgrf.car_precision(W_CHAIN, rho=0.5, tau=2.0)["Q"]
    assert q[0][0] == pytest.approx(2.0 * 1.0)
    assert q[1][1] == pytest.approx(2.0 * 2.0)
    assert q[0][1] == pytest.approx(-2.0 * 0.5)


def test_coclustering_is_symmetric_with_a_unit_diagonal():
    co = dpgrf.coclustering(DRAWS)
    assert co["symmetric"] and co["unit_diagonal"]
    assert co["matrix"][0][1] == pytest.approx(0.75)


def test_boundaries_are_the_adjacencies_that_cluster_apart():
    bp = dpgrf.boundary_probabilities(W_CHAIN, DRAWS, threshold=0.5)
    assert bp["boundaries"] == [(1, 2)]
    assert all(0.0 <= d["p_difference"] <= 1.0 for d in bp["ranked"])


def test_the_threshold_selects_and_does_not_rescale():
    lo = dpgrf.boundary_probabilities(W_CHAIN, DRAWS, threshold=0.2)
    hi = dpgrf.boundary_probabilities(W_CHAIN, DRAWS, threshold=0.99)
    assert lo["n_boundaries"] >= hi["n_boundaries"]
    assert [d["p_difference"] for d in lo["ranked"]] == \
        [d["p_difference"] for d in hi["ranked"]]


def test_the_dp_prior_produces_ties_among_regions():
    assert dpgrf.sample_labels(50, 1.0, seed=4)["n_clusters"] < 50


# ------------------------------------------------------------- poltrx
def test_the_c_m_squared_rule_grows_with_the_level():
    assert poltrx.level_parameters(3, 1.0)["alpha"] == 9.0
    assert poltrx.level_parameters(3, 1.0, "constant")["alpha"] == 1.0
    assert poltrx.level_parameters(3, 2.0, "linear")["alpha"] == 6.0


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        poltrx.level_parameters(1, 1.0, "quadratic_ish")


def test_the_regimes_are_named():
    assert (poltrx.continuity_regime("m_squared")["draws"]
            == "absolutely continuous")
    assert "DP" in poltrx.continuity_regime("constant")["draws"]


def test_the_binary_address_locates_the_point():
    pi = poltrx.partition_index(0.3, 3)
    assert pi["epsilon"] == (0, 1, 0)
    assert pi["interval"][0] == pytest.approx(0.25)
    assert pi["interval"][1] == pytest.approx(0.375)


@pytest.mark.parametrize("x", [-0.01, 1.5])
def test_a_point_outside_the_unit_interval_is_refused(x):
    with pytest.raises(ValueError):
        poltrx.partition_index(x, 2)


def test_the_tree_is_truncated_at_a_stated_level():
    t = poltrx.finite_tree(4, c=1.0, seed=11)
    assert t["levels"] == 4 and t["n_nodes"] == 15


def test_the_level_probabilities_sum_to_one():
    t = poltrx.finite_tree(5, c=1.0, seed=3)
    assert poltrx.tree_density(t)["total"] == pytest.approx(1.0,
                                                            abs=1e-12)


def test_a_set_probability_is_the_product_down_its_branch():
    t = poltrx.finite_tree(4, c=1.0, seed=11)
    p = poltrx.set_probability((0, 1), t)["probability"]
    assert p == pytest.approx(t["Y"][()] * (1.0 - t["Y"][(0,)]))


def test_asking_below_the_truncation_is_refused():
    with pytest.raises(ValueError):
        poltrx.set_probability((0, 1, 0), poltrx.finite_tree(2, seed=1))
