"""mrm_siu case-to-decision summaries: Kaplan-Meier as survival::survfit."""

import pandas as pd

from morie.mrm_siu import _km_summary, mrm_siu_case_to_decision_km


def test_km_summary_matches_survfit():
    # reference: survival::survfit(Surv(t, e) ~ 1): quantile(c(.25, .5, .75)) and
    # summary(rmean = "common")
    cases = [
        ([3, 5, 5, 8, 10, 12, 12, 15, 20, 22], [1, 1, 0, 1, 1, 0, 1, 1, 0, 1], [8, 12, 22], 13.2190476190476),
        ([1, 2, 3, 4], [1, 1, 1, 1], [1.5, 2.5, 3.5], 2.5),
        ([2, 4, 6, 8, 10, 12], [1, 1, 1, 0, 0, 0], [4, 9, None], 8.0),
        ([5, 5, 5, 7, 9], [1, 1, 0, 0, 0], [5, None, None], 7.4),
    ]
    for t, e, q_ref, rm_ref in cases:
        q, rm = _km_summary(t, e)
        for a, b in zip(q, q_ref):
            assert (b is None and a != a) or abs(a - b) < 1e-12
        assert abs(rm - rm_ref) < 1e-12


def test_case_to_decision_censors_open_cases():
    base = pd.Timestamp("2020-01-01")
    inc = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45]
    dec = [3, 15, 28, 55, None, 10, 30, 30, None, 60]
    d = pd.DataFrame(
        {
            "police_service": ["A"] * 5 + ["B"] * 5,
            "date_of_incident_iso": [str((base + pd.Timedelta(days=v)).date()) for v in inc],
            "date_of_director_decision_iso": [
                None if v is None else str((base + pd.Timedelta(days=v)).date()) for v in dec
            ],
        }
    )
    r = mrm_siu_case_to_decision_km(d, min_n=1)
    row = r.pooled.iloc[0]
    # survfit on these gaps: quartiles 5, 11.5, 25; rmean 13.1
    assert (row["p25_days"], row["median_days"], row["p75_days"]) == (5, 11.5, 25)
    assert row["mean_days"] == 13.1
    assert row["n_censored"] == 2
