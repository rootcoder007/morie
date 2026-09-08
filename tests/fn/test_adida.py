"""Tests for adida. Full anchor: wave3/anchor_intermittent.py."""
import pytest
from morie.fn import _array_core as np
from morie.fn.adida import (adida_forecast, aggregate_buckets,
                            disaggregate, temporal_combination,
                            zero_fraction)


@pytest.fixture(scope="module")
def demand():
    rng = np.random.default_rng(17)
    d = [0.0] * 96
    for i in range(0, 96, 4):
        d[i] = 5.0 + rng.standard_normal()
    return d


def test_aggregation_cuts_the_zero_fraction(demand):
    agg = aggregate_buckets(demand, 4)
    assert zero_fraction(agg) < zero_fraction(demand)
    assert len(agg) == 24
    assert sum(agg) == pytest.approx(sum(demand), abs=1e-9)
    assert len(aggregate_buckets(demand, 4, overlapping=True)) == 93


def test_disaggregation_sums_back_exactly():
    """A disaggregation that does not reconstitute is silently changing
    the total."""
    parts = disaggregate(20.0, 4)
    assert sum(parts) == pytest.approx(20.0, abs=1e-12)
    assert all(v == pytest.approx(5.0) for v in parts)
    prof = disaggregate(20.0, 4, profile=[3.0, 1.0, 0.0, 0.0])
    assert sum(prof) == pytest.approx(20.0, abs=1e-12)
    assert prof[0] == pytest.approx(15.0)
    with pytest.raises(ValueError):
        disaggregate(20.0, 4, profile=[-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        disaggregate(20.0, 4, profile=[0.0] * 4)
    with pytest.raises(ValueError):
        disaggregate(20.0, 4, profile=[1.0, 1.0])


def test_lead_time_aggregation_needs_no_disaggregation(demand):
    """The paper's recommendation: m = lead time makes the aggregate
    forecast BE lead-time demand."""
    r = adida_forecast(demand, 1, horizon=6, lead_time=6)
    assert r["lead_time_demand"] is not None
    assert sum(r["forecast"]) == pytest.approx(r["aggregate_forecast"],
                                               abs=1e-9)
    plain = adida_forecast(demand, 4, horizon=8)
    assert plain["disaggregation_sums_back"]


def test_temporal_combination_sits_within_its_members(demand):
    tc = temporal_combination(demand, [2, 4, 8], horizon=4)
    lo = min(p[0] for p in tc["per_level"])
    hi = max(p[0] for p in tc["per_level"])
    assert lo - 1e-9 <= tc["forecast"][0] <= hi + 1e-9
    assert len(tc["weights"]) == 3


def test_argument_checks(demand):
    with pytest.raises(ValueError):
        aggregate_buckets(demand, 200)
    with pytest.raises(ValueError):
        aggregate_buckets(demand, 0)
    with pytest.raises(ValueError):
        temporal_combination(demand, [4])
    with pytest.raises(ValueError):
        zero_fraction([])
