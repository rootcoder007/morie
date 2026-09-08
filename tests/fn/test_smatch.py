"""smatch -- Poisson form and sample size. Source: Whitaker, H. J.,
Farrington, C. P., Spiessens, B. & Musonda, P. (2006) Statistics in
Medicine 25, 1768-1797, doi:10.1002/sim.2302."""
import math

import pytest

from morie.fn.sccsno import sccs_fit
from morie.fn.smatch import (poisson_design, power,
                             relative_efficiency, sample_size,
                             sccs_poisson_fit)

RISK = [(0.0, 42.0)]


def series(n=150):
    """A non-degenerate series: events fall in BOTH the risk window
    and the control period, so the MLE is finite."""
    out = []
    for i in range(n):
        v = 20.0 + (i % 30)
        t = (v + 10.0) if (i % 3) else (v + 120.0)
        out.append({"start": 0.0, "end": 200.0, "exposure": v,
                    "events": [t]})
    return out


def test_poisson_fit_matches_the_conditional_fit():
    # no age bands: the two fits then agree to solver precision on a
    # well-conditioned series
    c = []
    for _ in range(120):
        c.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                  "events": [25.0]})
    for _ in range(180):
        c.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                  "events": [80.0]})
    a = sccs_fit(c, RISK, [])["log_ri"][0]
    b = sccs_poisson_fit(c, RISK, [])["log_ri"][0]
    assert a == pytest.approx(b, abs=1e-5)


def test_poisson_fit_also_hits_the_closed_form():
    import math as _m
    c = []
    for _ in range(120):
        c.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                  "events": [25.0]})
    for _ in range(180):
        c.append({"start": 0.0, "end": 100.0, "exposure": 20.0,
                  "events": [80.0]})
    got = sccs_poisson_fit(c, RISK, [])["relative_incidence"][0]
    assert got == pytest.approx((120 / 42.0) / (180 / 58.0), rel=1e-4)


def test_poisson_design_has_one_individual_column_per_case():
    d = poisson_design(series(20), RISK, [100.0])
    assert d["n_people"] == 20
    assert len(d["X"][0]) == d["n_risk"] + d["n_age"] - 1 + 20


def test_every_design_row_carries_exactly_one_person_indicator():
    d = poisson_design(series(10), RISK, [100.0])
    off = d["n_risk"] + d["n_age"] - 1
    assert all(sum(r[off:]) == 1.0 for r in d["X"])


def test_offsets_are_logs_of_positive_interval_lengths():
    d = poisson_design(series(5), RISK, [])
    assert all(math.exp(v) > 0.0 for v in d["offset"])


def test_sample_size_rho_matches_its_closed_form():
    s = sample_size(math.log(2.0), 0.1, 0.5)
    assert s["rho"] == pytest.approx(0.1 * 2.0 / (0.1 * 2.0 + 0.9))


def test_B_tends_to_one_as_the_effect_shrinks():
    assert sample_size(1e-4, 0.1, 0.5)["B"] == pytest.approx(1.0,
                                                             abs=1e-4)


def test_C_is_one_when_everyone_is_exposed():
    assert sample_size(math.log(2.0), 0.1, 1.0)["C"] == pytest.approx(
        1.0, abs=1e-13)


def test_a_rarer_exposure_needs_more_events():
    a = sample_size(math.log(2.0), 0.1, 0.5)["n_events"]
    b = sample_size(math.log(2.0), 0.1, 0.05)["n_events"]
    assert b > a


def test_a_bigger_effect_needs_fewer_events():
    a = sample_size(math.log(1.5), 0.1, 0.5)["n_events"]
    b = sample_size(math.log(4.0), 0.1, 0.5)["n_events"]
    assert b < a


def test_more_power_needs_more_events():
    a = sample_size(math.log(2.0), 0.1, 0.5, power=0.8)["n_events"]
    b = sample_size(math.log(2.0), 0.1, 0.5, power=0.95)["n_events"]
    assert b > a


def test_power_inverts_sample_size():
    n = sample_size(math.log(2.0), 0.1, 0.5, power=0.85)["n_events"]
    assert power(n, math.log(2.0), 0.1, 0.5)["power"] == pytest.approx(
        0.85, abs=1e-6)


def test_a_short_risk_period_keeps_efficiency_high():
    a = relative_efficiency(0.02, math.log(2.0))["efficiency"]
    b = relative_efficiency(0.5, math.log(2.0))["efficiency"]
    assert a > b


def test_a_zero_log_relative_incidence_is_refused():
    with pytest.raises(ValueError):
        sample_size(0.0, 0.1, 0.5)


def test_r_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        sample_size(0.5, 1.0, 0.5)


def test_zero_exposure_prevalence_is_refused():
    with pytest.raises(ValueError):
        sample_size(0.5, 0.1, 0.0)


def test_an_out_of_range_power_is_refused():
    with pytest.raises(ValueError):
        sample_size(0.5, 0.1, 0.5, power=1.0)


def test_a_design_with_no_events_is_refused():
    with pytest.raises(ValueError):
        poisson_design([{"start": 0.0, "end": 1.0, "exposure": 0.5,
                         "events": []}], RISK)
