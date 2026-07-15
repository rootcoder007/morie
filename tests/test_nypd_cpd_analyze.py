"""Smoke tests for the NYPD/CPD analysis aggregators on bundled samples."""

from morie import cpd_all_analyze, nypd_all_analyze


def test_nypd_analyze_all_runs_clean():
    results = nypd_all_analyze.analyze_all()
    assert results
    for name, res in results.items():
        assert not res.warnings, f"{name}: {res.warnings}"


def test_cpd_analyze_all_runs_clean():
    results = cpd_all_analyze.analyze_all()
    assert results
    for name, res in results.items():
        assert not res.warnings, f"{name}: {res.warnings}"


def test_nypd_felony_disparity_non_degenerate():
    di = nypd_all_analyze.analyze_all()["felony_race_disparity"].payload[
        "disparate_impact"
    ]
    rates = list(di["rates"].values())
    # bundled sample is a real 400-row slice; felony rates must vary by group
    assert len(set(rates)) > 1
    assert all(r == r for r in rates)  # no NaN


def test_cpd_disparity_uses_felony_outcome():
    di = cpd_all_analyze.analyze_all()["arrest_race_disparity"].payload[
        "disparate_impact"
    ]
    rates = list(di["rates"].values())
    # all-ones outcome would make every rate exactly 1.0
    assert len(set(rates)) > 1
    assert all(r == r for r in rates)
