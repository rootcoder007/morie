"""agroclim: indicators recomputed from their definitions on constructed series."""

import math

import pytest

from morie.fn.agroclim import (
    chill_accumulation,
    cold_spell_duration,
    growing_degree_days,
    growing_season_length,
    lai_from_savi,
    rainfall_adequacy,
)


def test_gdd_and_chill():
    r = growing_degree_days([20.0, 8.0, 35.0], [10.0, 2.0, 25.0], base=10.0)
    assert r.daily == [5.0, 0.0, 20.0] and r.cumulative == [5.0, 5.0, 25.0] and r.total == 25.0
    c = growing_degree_days([35.0], [5.0], base=10.0, upper=30.0)
    assert c.daily == [(30 + 10) / 2 - 10]
    temps = [0.0, 1.4, 1.5, 2.4, 2.5, 7.2, 9.1, 9.2, 12.4, 12.5, 15.9, 16.0, 18.0, 18.1, -3.0]
    ch = chill_accumulation(temps)
    assert ch.chill_hours == 6
    assert ch.utah_units == pytest.approx(0 + 0 + 0.5 + 0.5 + 1 + 1 + 1 + 0.5 + 0.5 + 0 + 0 - 0.5 - 0.5 - 1 + 0)


def test_lai_rainfall():
    s = [0.1, 0.2, 0.5, 0.686, 0.687]
    got = lai_from_savi(s)
    assert got[0] == 0.0 and got[-1] == 6.0
    assert got[1:4] == pytest.approx([-math.log((0.69 - v) / 0.59) / 0.91 for v in s[1:4]])
    assert lai_from_savi([0.686], cap=3.0) == [3.0]
    r = rainfall_adequacy([10.0, 60.0, 0.0], [50.0, 40.0, 30.0], kc=0.8)
    assert r.etc == pytest.approx([40.0, 32.0, 24.0])
    assert r.ratio == pytest.approx(70 / 96) and r.deficit == pytest.approx([30.0, 0.0, 24.0])


def test_growing_season_and_cold_spells():
    # season starts at the first 6-day run above 5 C (day 60) and ends before the first 6-day run below 5 C after 1 July
    T = [0.0] * 60 + [8.0] * 200 + [2.0] * 105
    assert growing_season_length(T) == 200
    short = [0.0] * 60 + [8.0] * 5 + [0.0] * 10 + [8.0] * 290
    assert growing_season_length(short) == 290
    early_cold = [0.0] * 50 + [9.0] * 50 + [0.0] * 10 + [9.0] * 255
    assert growing_season_length(early_cold) == 365 - 50  # cold run before 1 July does not end the season
    assert growing_season_length([0.0] * 365) == 0
    c = cold_spell_duration([0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0], [1.0] * 15)
    assert (c.csdi, c.spells) == (14, 2)
    assert cold_spell_duration([0.0] * 5, 1.0).csdi == 0
