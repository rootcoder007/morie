# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lakner's stock and flow measures, and MRM across the OTIS strata.

The anchors are Lakner's own worked numbers, not this implementation's
output, so they can fail: A Manual of Statistical Sampling Methods for
Corrections Planners (University of Illinois at Urbana-Champaign, 1976),
p.15-21.
"""

import pytest

import morie


# ---------------------------------------------------------------- measures

def test_adp_is_days_per_day():
    # p.15: 13,500 person-days over a year is 36.99 average daily
    assert morie.adp([13500]) == pytest.approx(13500 / 365, abs=1e-9)
    assert morie.adp([13500], t=365) == pytest.approx(36.98630137, abs=1e-8)


def test_alos_is_days_per_person():
    # p.16: 12,150 person-days over 2,700 releases is a 4.5 day stay
    assert morie.alos([12150], 2700) == pytest.approx(4.5, abs=1e-12)


def test_the_identity_closes():
    # eq 2.4, p.18: adp = N_a * alos / t, so N_a = adp * t / alos
    days, n = [13500.0], 2700.0
    a, l = morie.adp(days), morie.alos(days, n)
    assert morie.admissions(a, l) == pytest.approx(n, abs=1e-9)
    # p.20's worked figure: an adp of 25 and a stay of 5.214 days is 1,750
    assert morie.admissions(25, 25 * 365 / 1750) == pytest.approx(1750, abs=1e-6)


def test_person_days_from_periodic_counts():
    # eq 2.7, p.21: mean of the counts times the period
    assert morie.adp_from_counts([34935 / 255]) == pytest.approx(50005, abs=1e-6)
    assert morie.adp_from_counts([10, 20, 30], t=10) == pytest.approx(200)


def test_period_days_is_inclusive():
    assert morie.period_days("2024-01-01", "2024-12-31") == 366.0  # leap
    assert morie.period_days("2023-01-01", "2023-12-31") == 365.0
    assert morie.period_days("2024-03-01", "2024-03-01") == 1.0


def test_stay_summary_matches_r_conventions():
    x = [1.0, 2.0, 3.0, 4.0, 100.0]
    s = morie.stay_summary(x)
    assert s.n == 5
    assert s.total_days == pytest.approx(110.0)
    assert s.mean == pytest.approx(22.0)
    # Values taken from rmoriebricklayer::stay_summary(c(1,2,3,4,100)),
    # not from this implementation. R's sd() is the SAMPLE sd (n - 1).
    assert s.sd == pytest.approx(43.6176569751, abs=1e-9)
    assert s.se == pytest.approx(19.5064092031, abs=1e-9)
    assert s.lower == pytest.approx(-32.1584743520, abs=1e-9)
    assert s.upper == pytest.approx(76.1584743520, abs=1e-9)
    # R's default quantile is type 7
    assert s.median == pytest.approx(3.0)
    assert s.iqr == pytest.approx(2.0)
    assert s.lower < s.mean < s.upper


def test_a_single_observation_has_no_interval():
    s = morie.stay_summary([7.0])
    assert s.n == 1 and s.mean == 7.0
    assert s.sd is None and s.lower is None and s.upper is None


# -------------------------------------------------------------- stock/flow

def test_flow_and_stock_can_carry_opposite_signs():
    # THE SUBSTANCE. days = people x stay, so fewer people serving longer
    # stays can mean MORE detention days. Quoting the flow rate alone
    # reverses the finding, which is why both are returned.
    sf = morie.stock_flow(days=[100000.0, 109000.0], people=[1000.0, 760.0],
                          period=["FY2023", "FY2025"],
                          exposure=[14000000.0, 14000000.0])
    assert sf.people_change[-1] < 0      # fewer people
    assert sf.alos_change[-1] > 0        # longer stays
    assert sf.days_change[-1] > 0        # more days
    assert sf.flow_rate_change[-1] < 0
    assert sf.stock_rate_change[-1] > 0
    assert sf.opposite_signs() is True


def test_the_decomposition_is_exact():
    sf = morie.stock_flow(days=[100000.0, 109000.0], people=[1000.0, 760.0])
    p = sf.people_change[-1] / 100
    l = sf.alos_change[-1] / 100
    assert (1 + p) * (1 + l) - 1 == pytest.approx(sf.days_change[-1] / 100,
                                                  abs=1e-12)


def test_baseline_previous_differs_from_first():
    d, n = [100.0, 110.0, 121.0], [10.0, 10.0, 10.0]
    first = morie.stock_flow(days=d, people=n, baseline="first")
    prev = morie.stock_flow(days=d, people=n, baseline="previous")
    assert first.days_change[-1] == pytest.approx(21.0)
    assert prev.days_change[-1] == pytest.approx(10.0)
    assert prev.days_change[0] is None


def test_stock_flow_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        morie.stock_flow(days=[1.0, 2.0], people=[1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        morie.stock_flow(days=[1.0], people=[0.0])
    with pytest.raises(ValueError, match="baseline"):
        morie.stock_flow(days=[1.0], people=[1.0], baseline="middle")


# --------------------------------------------------------------------- MRM

def _otis(years=("FY2024", "FY2025"), n=(4, 3)):
    pd_rows, pl_rows = [], []
    for y, k in zip(years, n):
        for i in range(1, k + 1):
            days = 30.0 * i
            pd_rows.append({"EndFiscalYear": y,
                            "UniqueIndividual_ID": "%s-%d" % (y, i),
                            "TotalAggregatedDays_Segregation": days})
            # one person, three spells: spell lengths sum to the person's
            # days here, which is the FAVOURABLE case
            for _ in range(3):
                pl_rows.append({
                    "EndFiscalYear": y,
                    "UniqueIndividual_ID": "%s-%d" % (y, i),
                    "NumberConsecutiveDays_Segregation": days / 3,
                    "Number_Of_Placements": 1})
    totals = [{"EndFiscalYear": y, "NumberIndividuals_Segregation": k}
              for y, k in zip(years, n)]
    return pd_rows, pl_rows, totals


def test_mrm_reconciles_the_three_strata():
    pdr, plr, tot = _otis()
    r = morie.mrm_otis_stock_flow(pdr, placements=plr, totals=tot)
    assert [x["period"] for x in r.reconciliation] == ["FY2024", "FY2025"]
    assert r.reconciliation[0]["person_stratum_people"] == 4
    assert r.reconciliation[0]["person_stratum_days"] == pytest.approx(300.0)
    # person and aggregate strata agree here by construction
    assert r.strata_agree is True
    assert r.decomposition_exact is True
    assert r.reconciliation[0]["placements_per_person"] == pytest.approx(3.0)


def test_spell_days_are_reported_not_used_as_the_numerator():
    # A spell length is not an additive share of the year. On the
    # published release summing it understates person-days by about a
    # third, so the ratio is surfaced rather than silently applied.
    pdr, plr, _ = _otis()
    for row in plr:
        row["NumberConsecutiveDays_Segregation"] /= 2   # lossy spells
    r = morie.mrm_otis_stock_flow(pdr, placements=plr)
    assert r.reconciliation[0]["consecutive_over_person_days"] == \
        pytest.approx(0.5, abs=1e-12)
    # the measures still come off the PERSON stratum
    assert r.stock_flow.days[0] == pytest.approx(300.0)


def test_mrm_requires_the_columns_it_names():
    pdr, _, _ = _otis()
    with pytest.raises(ValueError, match="missing column"):
        morie.mrm_otis_stock_flow(pdr, days_col="NoSuchColumn")
    with pytest.raises(ValueError, match="missing column"):
        morie.mrm_otis_stock_flow(pdr, id_col="NoSuchColumn")
    with pytest.raises(ValueError, match="missing column"):
        morie.mrm_otis_stock_flow(pdr, placements=[{"EndFiscalYear": "FY2024"}])
    with pytest.raises(ValueError, match="rows of mappings"):
        morie.mrm_otis_stock_flow([1, 2, 3])
