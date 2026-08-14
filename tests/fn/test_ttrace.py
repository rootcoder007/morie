"""ttrace -- contact tracing and isolation. Source: Hellewell, J. et
al. (2020) The Lancet Global Health 8, e488-e496."""
import pytest

from morie.fn import _array_core as np
from morie.fn.ttrace import (effective_reproduction_number,
                             negbinom_offspring,
                             probability_of_control,
                             serial_interval_draw, simulate_outbreak)


def moments(R0, k, n=40000, seed=1):
    rng = np.random.default_rng(seed)
    d = [negbinom_offspring(R0, k, rng) for _ in range(n)]
    m = sum(d) / len(d)
    v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    return m, v


def test_offspring_mean_is_r0():
    m, _ = moments(2.5, 0.5)
    assert m == pytest.approx(2.5, abs=0.08)


def test_offspring_variance_is_r0_times_one_plus_r0_over_k():
    m, v = moments(1.5, 1.0)
    assert v == pytest.approx(1.5 * (1.0 + 1.5 / 1.0), rel=0.15)


def test_large_dispersion_gives_the_poisson_case():
    m, v = moments(3.0, 1e9)
    assert v == pytest.approx(m, rel=0.1)


def test_small_dispersion_gives_far_more_variance():
    _, v_lo = moments(2.5, 10.0)
    _, v_hi = moments(2.5, 0.1)
    assert v_hi > 3.0 * v_lo


def test_zero_r0_produces_no_offspring():
    rng = np.random.default_rng(0)
    assert all(negbinom_offspring(0.0, 1.0, rng) == 0
               for _ in range(50))


def test_serial_interval_without_presymptomatic_is_non_negative():
    rng = np.random.default_rng(0)
    vals = [serial_interval_draw(1.0, 5.0, rng,
                                 allow_presymptomatic=False)
            for _ in range(200)]
    assert all(v >= 0.0 for v in vals)


def test_r_eff_never_exceeds_r0():
    r = effective_reproduction_number(2.5, 4.7, 2.9, 3.8, 2.4,
                                      trace_prob=0.0, draws=4000,
                                      seed=1)
    assert r["R_eff"] <= 2.5 + 1e-9


def test_tracing_lowers_r_eff():
    a = effective_reproduction_number(2.5, 4.7, 2.9, 3.8, 2.4,
                                      trace_prob=0.0, draws=4000,
                                      seed=1)["R_eff"]
    b = effective_reproduction_number(2.5, 4.7, 2.9, 3.8, 2.4,
                                      trace_prob=1.0, draws=4000,
                                      seed=1)["R_eff"]
    assert b < a


def test_a_shorter_delay_lowers_r_eff():
    a = effective_reproduction_number(2.5, 4.7, 2.9, 9.0, 1.0,
                                      trace_prob=0.5, draws=4000,
                                      seed=2)["R_eff"]
    b = effective_reproduction_number(2.5, 4.7, 2.9, 1.0, 1.0,
                                      trace_prob=0.5, draws=4000,
                                      seed=2)["R_eff"]
    assert b < a


def test_subclinical_cases_survive_perfect_tracing():
    r = effective_reproduction_number(2.5, 4.7, 2.9, 0.0, 0.001,
                                      trace_prob=1.0,
                                      subclinical=0.5, draws=6000,
                                      seed=3)
    assert r["R_eff"] > 0.9 * 2.5 * 0.5


def test_a_subcritical_outbreak_is_controlled():
    r = probability_of_control(reps=30, seed=1, R0=0.3,
                               dispersion=0.5, n_initial=5,
                               trace_prob=0.0, max_weeks=10)
    assert r["probability_of_control"] > 0.9


def test_a_supercritical_untraced_outbreak_is_not():
    r = probability_of_control(reps=30, seed=1, R0=3.5,
                               dispersion=1.0, n_initial=40,
                               trace_prob=0.0, delay_mean=8.0,
                               max_weeks=12)
    assert r["probability_of_control"] < 0.3


def test_the_simulator_reports_its_control_definition():
    r = probability_of_control(reps=5, seed=1, R0=0.5, max_weeks=6)
    assert r["max_weeks"] == 6
    assert "max_cases" in r


def test_weekly_incidence_sums_to_at_most_the_total():
    out = simulate_outbreak(seed=2, R0=1.5, n_initial=5, max_weeks=8)
    assert sum(out["weekly"]) <= out["total_cases"]


def test_an_out_of_range_trace_probability_is_refused():
    with pytest.raises(ValueError):
        simulate_outbreak(trace_prob=-0.1)


def test_an_out_of_range_subclinical_fraction_is_refused():
    with pytest.raises(ValueError):
        simulate_outbreak(subclinical=1.5)


def test_a_non_positive_dispersion_is_refused():
    with pytest.raises(ValueError):
        negbinom_offspring(2.0, -1.0, np.random.default_rng(0))


def test_a_negative_r0_is_refused():
    with pytest.raises(ValueError):
        negbinom_offspring(-0.5, 1.0, np.random.default_rng(0))


def test_no_initial_cases_is_refused():
    with pytest.raises(ValueError):
        simulate_outbreak(n_initial=0)


def test_a_non_positive_serial_interval_sd_is_refused():
    with pytest.raises(ValueError):
        serial_interval_draw(5.0, 0.0, np.random.default_rng(0))
