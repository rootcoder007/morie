"""Partial-identification family: bnshrt, bndbye, bndpcb, bndsmw,
bnskmt, bnskt2.

Sources, all verified on page one of the local PDF:
  Honore & Tamer (2006) Econometrica 74(3), 611-629,
    doi:10.1111/j.1468-0262.2006.00676.x
  Moon & Schorfheide (2012) Econometrica 80(2), 755-782,
    doi:10.3982/ECTA8360
  Muller & Norets (2016) Econometrica 84(6), 2183-2213,
    doi:10.3982/ECTA14023
  Andrews & Shi, Cowles DP 1761R; Econometrica 81(2), 609-666 (2013)
  Card, Lee, Pei & Weber, NBER WP 18564; Econometrica 83(6),
    2453-2483 (2015)
"""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn.bndbye import (compare_sets, frequentist_confidence_set,
                             identified_set_interval, posterior_hpd)
from morie.fn.bndpcb import (bet_proof_interval, bet_violation,
                             coverage_by_region,
                             truncated_normal_interval)
from morie.fn.bndsmw import (S_function, cvm_statistic,
                             hypercube_instruments, weighted_moments)
from morie.fn.bnshrt import (identified_set, in_identified_set,
                             sequence_frequencies,
                             sequence_probabilities)
from morie.fn.bnskmt import compare_forms, ks_statistic
from morie.fn.bnskt2 import (covariate_kink_test, density_kink_test,
                             local_polynomial_slope, rkd_estimate)

X3 = [[1.0], [0.0], [1.0]]


# ---------------------------------------------------------- bnshrt
def test_sequence_probabilities_sum_to_one():
    p = sequence_probabilities([0.5], 0.8, X3, 0.2, 0)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
    assert len(p) == 8


def test_the_lag_coefficient_changes_the_distribution():
    a = sequence_probabilities([0.5], 1.0, X3, 0.0, 0)
    b = sequence_probabilities([0.5], -1.0, X3, 0.0, 0)
    assert max(abs(a[k] - b[k]) for k in a) > 0.01


def test_a_wrong_length_beta_is_refused():
    with pytest.raises(ValueError):
        sequence_probabilities([0.5, 0.2], 0.5, X3, 0.0, 0)


def test_non_binary_choices_are_refused():
    with pytest.raises(ValueError):
        sequence_frequencies([[0, 1, 2]])


def test_ragged_sequences_are_refused():
    with pytest.raises(ValueError):
        sequence_frequencies([[0, 1], [1, 0, 1]])


def panel(n=2000, seed=3, b=0.5, g=0.8):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        a = float(rng.normal(0.0, 1.0))
        y = 1 if float(rng.uniform()) < 0.5 else 0
        seq = []
        for t in range(3):
            p = 1.0 / (1.0 + math.exp(-(X3[t][0] * b + g * y + a)))
            y = 1 if float(rng.uniform()) < p else 0
            seq.append(y)
        rows.append(seq)
    return rows


def test_the_true_parameter_is_feasible():
    freq = sequence_frequencies(panel())
    AG = [-2.0 + 0.5 * i for i in range(9)]
    r = in_identified_set(freq, [0.5], 0.8, X3, AG, tol=0.03)
    assert r["feasible"]


def test_a_far_parameter_is_rejected():
    freq = sequence_frequencies(panel())
    AG = [-2.0 + 0.5 * i for i in range(9)]
    good = in_identified_set(freq, [0.5], 0.8, X3, AG, tol=0.03)
    bad = in_identified_set(freq, [0.5], -3.0, X3, AG, tol=0.03)
    assert bad["discrepancy"] > 5.0 * good["discrepancy"]


def test_the_mixing_weights_form_a_distribution():
    freq = sequence_frequencies(panel())
    AG = [-1.0, 0.0, 1.0]
    r = in_identified_set(freq, [0.5], 0.8, X3, AG)
    assert sum(r["weights"]) == pytest.approx(1.0, abs=1e-6)
    assert all(w >= -1e-12 for w in r["weights"])


def test_point_identification_fails():
    rows = panel()
    AG = [-2.0 + 0.5 * i for i in range(9)]
    s = identified_set(rows, X3, [0.3, 0.5, 0.7], [0.4, 0.8, 1.2],
                       AG, tol=0.03)
    assert s["n_feasible"] > 1
    assert not s["point_identified"]


# ---------------------------------------------------------- bndbye
def test_the_identified_set_is_centred_on_phi_hat():
    t = identified_set_interval(2.0, 0.5)
    assert (t["lower"], t["upper"]) == (1.5, 2.5)


def test_a_negative_half_width_is_refused():
    with pytest.raises(ValueError):
        identified_set_interval(1.0, -0.1)


def test_the_hpd_lies_inside_the_identified_set():
    c = compare_sets(1.0, 0.5, se_phi=0.1)
    assert c["hpd_inside_identified_set"]


def test_the_confidence_set_contains_the_identified_set():
    c = compare_sets(1.0, 0.5, se_phi=0.1)
    assert c["cs_contains_identified_set"]


def test_the_credible_set_is_smaller_than_the_confidence_set():
    c = compare_sets(1.0, 0.5, se_phi=0.1)
    assert c["width_ratio_hpd_over_cs"] < 1.0


def test_a_peaked_conditional_prior_shrinks_the_hpd():
    t = identified_set_interval(1.0, 0.5)
    g = [0.5 + i / 400.0 for i in range(401)]
    d = [math.exp(-0.5 * ((x - 0.7) / 0.1) ** 2) for x in g]
    z = sum(d) * (g[1] - g[0])
    peaked = {"grid": g, "density": [v / z for v in d]}
    a = posterior_hpd(t, conditional_prior=peaked)["width"]
    b = posterior_hpd(t)["width"]
    assert a < b


def test_a_noisier_estimate_widens_the_confidence_set():
    t = identified_set_interval(1.0, 0.5)
    a = frequentist_confidence_set(t, 0.05)["width"]
    b = frequentist_confidence_set(t, 0.40)["width"]
    assert b > a


def test_an_invalid_level_is_refused():
    t = identified_set_interval(1.0, 0.5)
    with pytest.raises(ValueError):
        posterior_hpd(t, level=0.0)


# ---------------------------------------------------------- bndpcb
def test_the_textbook_interval_can_be_empty():
    assert truncated_normal_interval(-2.5)["empty"]


def test_it_can_be_arbitrarily_short():
    assert 0.0 < truncated_normal_interval(-1.90)["width"] < 0.15


def test_an_ordinary_realisation_has_the_full_width():
    assert truncated_normal_interval(2.0)["width"] == pytest.approx(
        2 * 1.959963985, abs=1e-5)


def test_the_interval_is_marginally_valid():
    c = coverage_by_region(0.3, draws=6000, seed=1)
    assert c["marginal_coverage"] >= 0.93


def test_but_a_recognisable_subset_undercovers():
    b = bet_violation(0.2, draws=4000, seed=1)
    assert b["max_shortfall"] > 0.1


def test_the_bet_proof_interval_is_never_empty():
    r = bet_proof_interval(-3.0)
    assert not r["empty"] and r["width"] > 0.0


def test_the_bet_proof_interval_respects_its_floor():
    r = bet_proof_interval(-1.9, min_width=1.0)
    assert r["width"] >= 1.0 - 1e-12


def test_a_non_positive_floor_is_refused():
    with pytest.raises(ValueError):
        bet_proof_interval(0.0, min_width=-1.0)


# -------------------------------------------------- bndsmw / bnskmt
def test_S_is_zero_when_no_inequality_is_violated():
    assert S_function([1.0, 0.5, 3.0]) == 0.0


def test_S_squares_the_violation():
    assert S_function([-2.0]) == pytest.approx(4.0)


def test_the_max_form_takes_the_worst_violation():
    assert S_function([-1.0, -3.0], form="max") == pytest.approx(9.0)


def test_the_sum_form_accumulates_violations():
    assert S_function([-1.0, -3.0], form="sum") == pytest.approx(10.0)


def test_equalities_are_penalised_in_both_directions():
    assert S_function([2.0], n_equality=1) == pytest.approx(4.0)


def test_an_unknown_form_is_refused():
    with pytest.raises(ValueError):
        S_function([1.0], form="ks")


def moments(seed=2, n=200, mean=0.5):
    rng = np.random.default_rng(seed)
    X = [[float(rng.uniform())] for _ in range(n)]
    m = [[mean + 0.1 * float(rng.normal())] for _ in range(n)]
    return X, m


def test_instrument_weights_are_non_negative():
    X, _ = moments()
    inst = hypercube_instruments(X, n_levels=2)
    assert all(v >= 0.0 for g in inst["instruments"] for v in g)


def test_a_satisfied_inequality_gives_a_zero_statistic():
    X, m = moments(mean=0.5)
    assert cvm_statistic(m, hypercube_instruments(X, 2))["statistic"] \
        == 0.0


def test_a_violated_inequality_gives_a_large_statistic():
    X, m = moments(mean=-0.5)
    assert cvm_statistic(m, hypercube_instruments(X, 2))["statistic"] \
        > 10.0


def test_the_supremum_is_never_below_the_average():
    X, m = moments(mean=-0.2)
    inst = hypercube_instruments(X, 2)
    c = compare_forms(m, inst)
    assert c["ks"] >= c["cvm"] - 1e-9


def test_a_negative_weight_is_refused():
    X, m = moments()
    with pytest.raises(ValueError):
        weighted_moments(m, [-1.0] * len(m))


def test_a_measure_not_summing_to_one_is_refused():
    X, m = moments()
    inst = hypercube_instruments(X, 2)
    with pytest.raises(ValueError):
        cvm_statistic(m, inst,
                      weights=[1.0] * inst["n_instruments"])


# ---------------------------------------------------------- bnskt2
def kink_data(n=3000, seed=8, tau=2.0, dslope=-1.0):
    rng = np.random.default_rng(seed)
    V, Y, B = [], [], []
    for _ in range(n):
        v = -1.0 + 2.0 * float(rng.uniform())
        b = 1.0 * v if v < 0.0 else (1.0 + dslope) * v
        V.append(v)
        B.append(b)
        Y.append(3.0 + 0.5 * v + tau * b
                 + float(rng.normal(0.0, 0.1)))
    return V, Y, B


def test_sharp_rkd_recovers_the_planted_effect():
    V, Y, _ = kink_data()
    r = rkd_estimate(V, Y, 0.0, 0.5, order=2,
                     policy_slope_change=-1.0)
    assert r["tau"] == pytest.approx(2.0, abs=0.2)


def test_fuzzy_rkd_recovers_the_policy_kink():
    V, Y, B = kink_data()
    r = rkd_estimate(V, Y, 0.0, 0.5, order=2, B=B, fuzzy=True)
    assert r["policy_kink"] == pytest.approx(-1.0, abs=0.1)


def test_a_zero_policy_kink_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        rkd_estimate(V, Y, 0.0, 0.5, policy_slope_change=0.0)


def test_sharp_rkd_without_the_policy_slope_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        rkd_estimate(V, Y, 0.0, 0.5)


def test_fuzzy_rkd_without_treatment_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        rkd_estimate(V, Y, 0.0, 0.5, fuzzy=True)


def test_a_clean_density_is_smooth():
    V, _, _ = kink_data()
    assert density_kink_test(V, 0.0, 0.5, n_bins=20)["smooth"]


def test_a_manipulated_density_is_not_smooth():
    V, _, _ = kink_data()
    rng = np.random.default_rng(4)
    Vm = [(-0.02 * float(rng.uniform())
           if (0.0 < v < 0.25 and float(rng.uniform()) < 0.8) else v)
          for v in V]
    assert not density_kink_test(Vm, 0.0, 0.5, n_bins=20)["smooth"]


def test_an_unaffected_covariate_shows_no_kink():
    V, _, _ = kink_data()
    rng = np.random.default_rng(6)
    Z = [0.2 * v + float(rng.normal(0.0, 0.1)) for v in V]
    r = covariate_kink_test(V, Z, 0.0, 0.5, order=2)
    assert abs(r["slope_change"]) < 0.4


def test_a_non_positive_bandwidth_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        local_polynomial_slope(V, Y, 0.0, -0.1)


def test_an_unknown_kernel_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        local_polynomial_slope(V, Y, 0.0, 0.5, kernel="gaussian")


def test_an_unknown_side_is_refused():
    V, Y, _ = kink_data()
    with pytest.raises(ValueError):
        local_polynomial_slope(V, Y, 0.0, 0.5, side="both")
