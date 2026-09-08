"""snmtst -- honest DiD sensitivity. Source: Rambachan & Roth (2023)
Review of Economic Studies 90(5), 2555-2591."""
import pytest

from morie.fn.snmtst import (breakdown_value, fixed_length_ci,
                             identified_set, sensitivity_curve)

BETA = [-0.4, -0.2, 0.0, 1.0, 1.2]      # 3 pre, 2 post


def test_sd_at_zero_is_a_point():
    s = identified_set(BETA, 3, 2, M=0.0, family="SD")
    assert s["width"] == pytest.approx(0.0, abs=1e-14)


def test_sd_at_zero_is_linear_extrapolation():
    s = identified_set(BETA, 3, 2, M=0.0, family="SD")
    assert s["estimate"] == pytest.approx(1.0 - 0.2, abs=1e-12)


def test_sd_closed_form_matches_the_brute_force_recursion():
    a = identified_set(BETA, 3, 2, M=0.3, family="SD",
                       l_vec=[1.0, -1.0])
    b = identified_set(BETA, 3, 2, M=0.3, family="SD",
                       l_vec=[1.0, -1.0], grid=31)
    assert a["lower"] == pytest.approx(b["lower"], abs=1e-9)
    assert a["upper"] == pytest.approx(b["upper"], abs=1e-9)


def test_sd_half_width_is_M_times_the_first_coefficient():
    s = identified_set(BETA, 3, 2, M=0.25, family="SD")
    assert s["width"] / 2.0 == pytest.approx(0.25, abs=1e-12)


def test_width_is_monotone_in_M():
    w = sensitivity_curve(BETA, 3, 2, [0.0, 0.1, 0.5, 2.0])["width"]
    assert all(w[i] <= w[i + 1] + 1e-12 for i in range(len(w) - 1))


def test_rm_half_width_is_M_times_the_largest_pre_change():
    s = identified_set(BETA, 3, 2, M=1.0, family="RM",
                       l_vec=[1.0, 0.0])
    assert s["width"] / 2.0 == pytest.approx(0.2, abs=1e-12)


def test_rm_at_zero_is_a_point_at_beta_post():
    s = identified_set(BETA, 3, 2, M=0.0, family="RM")
    assert s["width"] == pytest.approx(0.0, abs=1e-14)
    assert s["estimate"] == pytest.approx(1.0, abs=1e-12)


def test_breakdown_puts_the_lower_bound_on_zero():
    b = breakdown_value(BETA, 3, 2, family="SD")
    s = identified_set(BETA, 3, 2, M=b["breakdown"], family="SD")
    assert abs(s["lower"]) < 1e-7


def test_a_failing_conclusion_has_a_breakdown_of_zero():
    b = breakdown_value([-0.4, -0.2, 0.0, 0.1, 0.1], 3, 2)
    assert b["breakdown"] == 0.0


def test_confidence_set_contains_the_identified_set():
    c = fixed_length_ci(BETA, 0.25, 3, 2, M=0.1)
    assert c["lower"] <= c["identified_lower"]
    assert c["upper"] >= c["identified_upper"]


def test_negative_M_is_refused():
    with pytest.raises(ValueError):
        identified_set(BETA, 3, 2, M=-0.1)


def test_wrong_coefficient_count_is_refused():
    with pytest.raises(ValueError):
        identified_set(BETA, 3, 3)


def test_unknown_family_is_refused():
    with pytest.raises(ValueError):
        identified_set(BETA, 3, 2, family="RM2")


def test_single_pre_period_is_refused_under_sd():
    with pytest.raises(ValueError):
        identified_set([0.0, 1.0], 1, 1, M=0.1)


def test_target_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError):
        identified_set(BETA, 3, 2, l_vec=[1.0])


def test_a_level_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        fixed_length_ci(BETA, 0.25, 3, 2, level=1.5)
