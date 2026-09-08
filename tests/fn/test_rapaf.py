"""rapaf -- adjusted PAF. Source: Bruzzi, P., Green, S. B., Byar,
D. P., Brinton, L. A. & Schairer, C. (1985) American Journal of
Epidemiology 122(5), 904-914 (the article prints no DOI)."""
import pytest

from morie.fn.rapaf import (ar_confidence_interval, levin_ar,
                            partial_ar, population_attributable_risk,
                            rate_ratios_from_logit)


@pytest.mark.parametrize("p,R", [(0.3, 2.0), (0.1, 5.0), (0.5, 1.4),
                                 (0.8, 3.3), (0.05, 10.0)])
def test_single_factor_equals_levin_exactly(p, R):
    rho1 = p * R / (1.0 - p + p * R)
    ar = population_attributable_risk([1.0 - rho1, rho1],
                                      [1.0, R])["ar"]
    assert ar == pytest.approx(levin_ar(p, R), abs=1e-13)


def test_no_excess_risk_gives_exactly_zero():
    assert population_attributable_risk([3.0, 7.0],
                                        [1.0, 1.0])["ar"] == 0.0


def test_ar_is_the_formula_it_claims_to_be():
    ar = population_attributable_risk([40.0, 60.0], [1.0, 4.0])["ar"]
    assert ar == pytest.approx(1.0 - (0.4 / 1.0 + 0.6 / 4.0),
                               abs=1e-15)


def test_ar_depends_only_on_case_proportions():
    a = population_attributable_risk([40.0, 60.0], [1.0, 4.0])["ar"]
    b = population_attributable_risk([400.0, 600.0], [1.0, 4.0])["ar"]
    assert a == pytest.approx(b, abs=1e-15)


def test_ar_never_reaches_one():
    assert population_attributable_risk([1.0, 1e6],
                                        [1.0, 1e9])["ar"] < 1.0


def test_the_control_distribution_is_never_used():
    r = population_attributable_risk([40.0, 60.0], [1.0, 4.0])
    assert r["uses_control_distribution"] is False


def test_a_bigger_rate_ratio_raises_the_ar():
    a = population_attributable_risk([50.0, 50.0], [1.0, 2.0])["ar"]
    b = population_attributable_risk([50.0, 50.0], [1.0, 8.0])["ar"]
    assert b > a


def test_partial_ars_do_not_add_up_to_the_joint_one():
    cases, rrs = [40.0, 30.0, 20.0, 10.0], [1.0, 2.0, 3.0, 6.0]
    joint = population_attributable_risk(cases, rrs)["ar"]
    a = partial_ar(cases, rrs, [0, 0, 2, 2])["ar"]
    b = partial_ar(cases, rrs, [0, 1, 0, 1])["ar"]
    assert joint < a + b
    assert a <= joint and b <= joint


def test_a_partial_ar_with_an_identity_map_is_zero():
    cases, rrs = [40.0, 30.0, 30.0], [1.0, 2.0, 3.0]
    assert partial_ar(cases, rrs, [0, 1, 2])["ar"] == pytest.approx(
        0.0, abs=1e-15)


def test_logit_rate_ratios_reference_the_first_stratum():
    r = rate_ratios_from_logit([10.0, 20.0], [30.0, 20.0],
                               [[0.0], [1.0]])
    assert r["rate_ratios"][0] == pytest.approx(1.0, abs=1e-12)


def test_a_saturated_logit_reproduces_the_cell_odds_ratios():
    ca = [100.0, 200.0, 150.0, 600.0]
    co = [400.0, 400.0, 300.0, 300.0]
    des = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
           [1.0, 1.0, 1.0]]
    got = rate_ratios_from_logit(ca, co, des)["rate_ratios"]
    want = [(ca[j] / co[j]) / (ca[0] / co[0]) for j in range(4)]
    for j in range(4):
        assert got[j] == pytest.approx(want[j], rel=1e-4)


def test_the_monte_carlo_interval_brackets_the_estimate():
    ci = ar_confidence_interval([40.0, 60.0], [1.0, 4.0],
                                [0.0, 0.2], draws=2000, seed=1)
    assert ci["lower"] < ci["estimate"] < ci["upper"]


def test_zero_standard_errors_give_a_degenerate_interval():
    ci = ar_confidence_interval([40.0, 60.0], [1.0, 4.0],
                                [0.0, 0.0], draws=200, seed=1)
    assert ci["upper"] - ci["lower"] == pytest.approx(0.0, abs=1e-12)


def test_a_non_positive_rate_ratio_is_refused():
    with pytest.raises(ValueError):
        population_attributable_risk([1.0, 1.0], [1.0, -2.0])


def test_a_negative_case_count_is_refused():
    with pytest.raises(ValueError):
        population_attributable_risk([-1.0, 2.0], [1.0, 2.0])


def test_no_cases_is_refused():
    with pytest.raises(ValueError):
        population_attributable_risk([0.0, 0.0], [1.0, 2.0])


def test_a_single_stratum_is_refused():
    with pytest.raises(ValueError):
        population_attributable_risk([5.0], [1.0])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        population_attributable_risk([1.0, 2.0], [1.0])


def test_a_prevalence_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        levin_ar(-0.1, 2.0)


def test_an_out_of_range_baseline_map_is_refused():
    with pytest.raises(ValueError):
        partial_ar([1.0, 1.0], [1.0, 2.0], [0, 5])
