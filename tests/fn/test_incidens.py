"""Verification tests for incidens.incidence_rate.

The incidence rate is cases divided by person-time, reported with an
exact Poisson interval. The expected rate is recomputed in the test
body and the interval is checked against the properties an exact
Poisson interval must have.
"""

import math

import pytest

from morie.fn.incidens import incidence_rate


def test_rate_is_cases_over_person_time():
    for d, pt in ((12, 480.0), (3, 1000.0), (57, 2350.5)):
        res = incidence_rate(d, pt)
        assert float(res.estimate) == pytest.approx(d / pt, rel=1e-12)


def test_interval_contains_the_rate_and_is_ordered():
    res = incidence_rate(12, 480.0, confidence=0.95)
    lo, hi = float(res.ci_lower), float(res.ci_upper)
    assert 0.0 <= lo <= float(res.estimate) <= hi


def test_no_cases_gives_a_zero_lower_bound_and_a_positive_upper_bound():
    res = incidence_rate(0, 500.0)
    assert float(res.estimate) == pytest.approx(0.0, abs=1e-15)
    assert float(res.ci_lower) == pytest.approx(0.0, abs=1e-15)
    assert float(res.ci_upper) > 0.0


def test_more_person_time_at_the_same_rate_narrows_the_interval():
    small = incidence_rate(10, 100.0)
    large = incidence_rate(100, 1000.0)
    assert float(small.estimate) == pytest.approx(float(large.estimate), rel=1e-12)
    width_small = float(small.ci_upper) - float(small.ci_lower)
    width_large = float(large.ci_upper) - float(large.ci_lower)
    assert width_large < width_small


def test_a_wider_confidence_level_gives_a_wider_interval():
    narrow = incidence_rate(12, 480.0, confidence=0.80)
    wide = incidence_rate(12, 480.0, confidence=0.99)
    assert float(wide.ci_upper) - float(wide.ci_lower) > \
        float(narrow.ci_upper) - float(narrow.ci_lower)
