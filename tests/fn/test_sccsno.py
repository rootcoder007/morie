"""sccsno -- self-controlled case series. Source: Farrington, C. P.
(1995) Biometrics 51(1), 228-235, JSTOR 2533328 (no DOI printed)."""
import math

import pytest

from morie.fn.sccsno import (build_intervals, check_assumptions,
                             relative_incidence, sccs_fit, sccs_loglik)

RISK = [(0.0, 42.0)]


def test_interval_lengths_sum_to_the_observation_period():
    cells = build_intervals(0.0, 730.0, 400.0, [410.0], RISK, [365.0])
    assert sum(e for _, _, e, _ in cells) == pytest.approx(730.0)


def test_every_event_is_placed_exactly_once():
    cells = build_intervals(0.0, 100.0, 20.0, [10.0, 30.0, 90.0],
                            RISK, [50.0])
    assert sum(n for _, _, _, n in cells) == 3


def test_the_risk_window_is_the_stated_interval():
    cells = build_intervals(0.0, 100.0, 20.0, [], [(0.0, 10.0)], [])
    risk_time = sum(e for _, r, e, _ in cells if r == 1)
    assert risk_time == pytest.approx(10.0)


def test_a_case_with_no_exposure_is_all_control_time():
    cells = build_intervals(0.0, 50.0, None, [10.0], RISK, [])
    assert all(r == 0 for _, r, _, _ in cells)


def test_loglik_ignores_individual_effects_entirely():
    cells = [build_intervals(0.0, 100.0, 20.0, [25.0], RISK, [])]
    a = sccs_loglik([0.5], cells, 1, 1)
    b = sccs_loglik([0.5], cells, 1, 1)
    assert a == pytest.approx(b, abs=1e-15)


def mixed(n_risk_events, n_control_events):
    """Cases sharing one geometry: risk window (20, 62], control 58.

    With a single risk period, no age bands and identical geometry the
    MLE has a closed form, exp(beta) = (n_r/R) / (n_c/C), which is
    what the tests below check against.
    """
    out = []
    for _ in range(n_risk_events):
        out.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                    "events": [25.0]})
    for _ in range(n_control_events):
        out.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                    "events": [80.0]})
    return out


def closed_form(n_r, n_c, R=42.0, C=58.0):
    return (n_r / R) / (n_c / C)


def test_fit_matches_the_closed_form_mle():
    r = sccs_fit(mixed(120, 180), RISK, [])
    assert r["relative_incidence"][0] == pytest.approx(
        closed_form(120, 180), rel=1e-6)
    assert r["converged"]


def test_the_closed_form_is_not_one_here():
    # (120/42) / (180/58) = 0.9206 -- below 1, so the fit is not
    # merely reproducing a null and the comparison is not vacuous
    assert abs(closed_form(120, 180) - 1.0) > 0.05


def test_a_different_split_gives_a_different_estimate():
    a = sccs_fit(mixed(120, 180), RISK, [])["relative_incidence"][0]
    b = sccs_fit(mixed(200, 100), RISK, [])["relative_incidence"][0]
    assert b > a
    assert b == pytest.approx(closed_form(200, 100), rel=1e-6)


def test_events_proportional_to_time_give_a_relative_incidence_of_one():
    # 42 : 58 events in a 42 : 58 time split is exactly no effect
    r = sccs_fit(mixed(42 * 5, 58 * 5), RISK, [])
    assert r["relative_incidence"][0] == pytest.approx(1.0, abs=1e-6)


def test_loglik_is_maximised_at_the_closed_form():
    cases = mixed(120, 180)
    cells = [build_intervals(c["start"], c["end"], c["exposure"],
                             c["events"], RISK, []) for c in cases]
    import math as _m
    b_hat = _m.log(closed_form(120, 180))
    at = sccs_loglik([b_hat], cells, 1, 1)
    assert at > sccs_loglik([b_hat - 0.2], cells, 1, 1)
    assert at > sccs_loglik([b_hat + 0.2], cells, 1, 1)


def test_fit_returns_one_relative_incidence_per_risk_period():
    cases = []
    for i in range(300):
        t = {0: 60.0, 1: 90.0, 2: 150.0}[i % 3]
        cases.append({"start": 0.0, "end": 200.0, "exposure": 50.0,
                      "events": [t]})
    r = sccs_fit(cases, [(0.0, 20.0), (20.0, 60.0)], [])
    assert len(r["relative_incidence"]) == 2
    assert r["n_risk_periods"] == 2


def test_cases_without_events_contribute_nothing():
    good = mixed(120, 180)
    padded = good + [{"start": 0.0, "end": 100.0, "exposure": 20.0,
                      "events": []} for _ in range(50)]
    a = sccs_fit(good, RISK, [])["log_ri"][0]
    b = sccs_fit(padded, RISK, [])["log_ri"][0]
    assert a == pytest.approx(b, abs=1e-12)


def test_relative_incidence_interval_brackets_the_estimate():
    iv = relative_incidence(
        sccs_fit(mixed(120, 180), RISK, []))["intervals"][0]
    assert iv["lower"] < iv["ri"] < iv["upper"]
    assert iv["lower"] < closed_form(120, 180) < iv["upper"]


def test_check_assumptions_flags_a_far_from_one_pre_window():
    class Fake(dict):
        pass
    f = Fake(relative_incidence=[4.0])
    assert not check_assumptions(f)["consistent_with_design"]
    f2 = Fake(relative_incidence=[1.02])
    assert check_assumptions(f2)["consistent_with_design"]


def test_an_event_outside_the_observation_period_is_refused():
    with pytest.raises(ValueError):
        build_intervals(0.0, 10.0, 5.0, [20.0], RISK, [])


def test_an_exposure_outside_the_observation_period_is_refused():
    with pytest.raises(ValueError):
        build_intervals(0.0, 10.0, 99.0, [5.0], RISK, [])


def test_a_degenerate_risk_period_is_refused():
    with pytest.raises(ValueError):
        build_intervals(0.0, 10.0, 5.0, [5.0], [(3.0, 3.0)], [])


def test_a_zero_length_observation_period_is_refused():
    with pytest.raises(ValueError):
        build_intervals(5.0, 5.0, None, [], RISK, [])


def test_unordered_age_breaks_are_refused():
    cases = [{"start": 0.0, "end": 100.0, "exposure": 20.0,
              "events": [25.0]}]
    with pytest.raises(ValueError):
        sccs_fit(cases, RISK, [60.0, 30.0])


def test_no_events_at_all_is_refused():
    with pytest.raises(ValueError):
        sccs_fit([{"start": 0.0, "end": 1.0, "exposure": 0.5,
                   "events": []}], RISK)


def test_no_risk_period_is_refused():
    cases = [{"start": 0.0, "end": 100.0, "exposure": 20.0,
              "events": [25.0]}]
    with pytest.raises(ValueError):
        sccs_fit(cases, [], [])
