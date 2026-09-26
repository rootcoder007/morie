"""Lakner (1976) stock and flow measures, anchored on the manual's worked examples.

Mirrors the R arm's test-mrm_stockflow_lakner.R (ported from bricklayer).
"""

import pytest

from morie import mrm_stockflow as S


def test_lakner_worked_examples():
    assert S.adp(13500) == 13500 / 365  # p.15
    assert round(S.adp(13500)) == 37
    assert S.adp([5000, 8500]) == S.adp(13500)
    assert S.alos(12150, 2700) == 4.5  # p.17
    implied = 25 * 365 / 1750  # p.20
    assert round(implied, 1) == 5.2
    assert S.admissions(25, implied) == pytest.approx(1750, rel=1e-15)
    assert S.adp_from_counts([34935 / 255] * 255) == pytest.approx(50005, rel=1e-15)  # p.21
    assert S.period_days("2024-01-01", "2024-12-31") == 366
    assert S.period_days("2023-01-01", "2023-12-31") == 365


def test_stock_and_flow_carry_opposite_signs():
    sf = S.stock_flow(
        days=[115674, 126121], people=[12647, 9608], period=["2023", "2025"], exposure=[15495050, 16256538]
    )
    assert sf.people_change[1] < 0 < sf.alos_change[1]
    assert sf.flow_rate_change[1] < 0 < sf.stock_rate_change[1]
    p, s = sf.people_change[1] / 100, sf.alos_change[1] / 100
    assert abs((1 + p) * (1 + s) - 1 - sf.days_change[1] / 100) < 1e-14
    assert sf.adp[1] == 126121 / 365
    assert sf.stock_rate[1] == pytest.approx(1e5 * (126121 / 365) / 16256538, rel=1e-15)
    assert sf.flow_rate[1] == pytest.approx(1e5 * 9608 / 16256538, rel=1e-15)
    # the OTIS reversal: flow -27.6%, stock +3.9%
    assert round(sf.flow_rate_change[1], 1) == -27.6
    assert round(sf.stock_rate_change[1], 1) == 3.9


def test_stay_summary_matches_the_r_arm():
    st = S.stay_summary([1] * 40 + [3] * 30 + [10] * 20 + [60, 90, 120])
    assert (st.n, st.total_days, st.median, st.iqr, st.max) == (93, 600, 3, 2, 120)
    assert abs(st.mean - 600 / 93) < 1e-14
    # R: t interval on the mean
    assert abs(st.lower - 3.08811278581036) < 1e-12
    assert abs(st.upper - 9.81511302064125) < 1e-12
