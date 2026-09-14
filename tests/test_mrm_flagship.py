# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MRM flagship: load with provenance, reconcile, report.

Parity with rmorie's R/mrm_flagship.R. The reconciliation anchors are
rmorie's own documented example, so they can fail.
"""

import pytest

import morie


# ------------------------------------------------------------ reconciliation

def test_reconcile_matches_rmories_documented_example():
    # From morie_mrm_reconcile's Rd:
    #   a <- data.frame(id = 1:5, y = c(1, 2, 3, 4, 5))
    #   b <- data.frame(id = c(1:4, 9), y = c(1, 2, 3.5, 4, 9))
    #   r$match_rate  -> 0.8 ; nrow(r$conflicts) -> 1
    a = [{"id": i, "y": float(i)} for i in range(1, 6)]
    b = [{"id": i, "y": v}
         for i, v in zip([1, 2, 3, 4, 9], [1.0, 2.0, 3.5, 4.0, 9.0])]
    r = morie.mrm_reconcile(a, b, keys="id", compare="y")
    assert r.match_rate == pytest.approx(0.8)
    assert len(r.conflicts) == 1
    assert len(r.matched) == 4
    assert len(r.unmatched_primary) == 1
    assert len(r.unmatched_secondary) == 1
    c = r.conflicts[0]
    assert c["key"] == "3" and c["field"] == "y"
    assert c["primary_value"] == "3.0" and c["secondary_value"] == "3.5"


def test_numeric_tolerance_forgives_bounded_disagreement():
    a = [{"id": 1, "y": 20.0}]
    b = [{"id": 1, "y": 20.4}]
    assert len(morie.mrm_reconcile(a, b, keys="id", compare="y").conflicts) == 1
    tol = morie.mrm_reconcile(a, b, keys="id", compare="y",
                              numeric_tolerance=0.5)
    assert tol.conflicts == []
    # tolerance applies to numerics only; a string mismatch is still a
    # conflict however wide the slack
    a2 = [{"id": 1, "lab": "r"}]
    b2 = [{"id": 1, "lab": "X"}]
    assert len(morie.mrm_reconcile(a2, b2, keys="id", compare="lab",
                                   numeric_tolerance=99).conflicts) == 1


def test_reconcile_joins_on_several_keys():
    a = [{"id": 1, "site": "A", "y": 1.0}, {"id": 1, "site": "B", "y": 2.0}]
    b = [{"id": 1, "site": "B", "y": 2.0}]
    r = morie.mrm_reconcile(a, b, keys=["id", "site"], compare="y")
    assert len(r.matched) == 1
    assert r.match_rate == pytest.approx(0.5)
    assert r.unmatched_primary[0]["site"] == "A"


def test_shared_columns_are_kept_from_both_sides():
    a = [{"id": 1, "y": 1.0}]
    b = [{"id": 1, "y": 2.0}]
    r = morie.mrm_reconcile(a, b, keys="id")
    row = r.matched[0]
    assert row["y.primary"] == 1.0 and row["y.secondary"] == 2.0


def test_a_missing_key_column_is_an_error():
    with pytest.raises(ValueError, match="missing column"):
        morie.mrm_reconcile([{"id": 1}], [{"other": 1}], keys="id")
    with pytest.raises(ValueError, match="at least one column"):
        morie.mrm_reconcile([{"id": 1}], [{"id": 1}], keys=[])


def test_no_overlap_reports_a_zero_rate_rather_than_failing():
    r = morie.mrm_reconcile([{"id": 1}], [{"id": 2}], keys="id")
    assert r.match_rate == 0.0
    assert r.matched == [] and len(r.unmatched_primary) == 1


# ------------------------------------------------------------------ loading

def test_provenance_records_the_snapshot_not_just_the_name():
    d = morie.mrm_load_si_dataset("otis_b01")
    p = d.provenance
    assert p["name"] == "otis_b01"
    assert p["n_rows"] > 0 and p["n_cols"] > 0
    assert len(p["sha256"]) == 64          # hex sha-256
    assert p["loaded_at"].endswith("Z")
    # the checksum is of the DATA, so loading twice agrees
    assert morie.mrm_load_si_dataset("otis_b01").provenance["sha256"] == \
        p["sha256"]


def test_reconcile_accepts_a_loaded_dataset_directly():
    d = morie.mrm_load_si_dataset("otis_b01")
    r = morie.mrm_reconcile(d, d, keys=list(d.data.columns)[:1])
    assert r.match_rate == pytest.approx(1.0)


# ---------------------------------------------------------------- reporting

class _Effect:
    results = [
        {"method": "ipw ate", "estimate": 0.812, "std_error": 0.101,
         "ci_lower": 0.614, "ci_upper": 1.010, "p_adjusted": 0.0004},
        {"method": "aipw", "estimate": 0.795, "std_error": 0.098,
         "ci_lower": 0.603, "ci_upper": 0.987, "p_adjusted": 0.02},
    ]
    consensus = {"estimate": 0.803, "std_error": 0.070}


def test_report_renders_every_section_it_is_given():
    d = morie.mrm_load_si_dataset("otis_b01")
    r = morie.mrm_reconcile(d, d, keys=list(d.data.columns)[:1])
    out = morie.mrm_report(effect=_Effect(), reconciliation=r, dataset=d)
    assert "Multilevel Reconciliation Methodology" in out
    assert "Source" in out and "sha256" in out
    assert "Reconciliation" in out and "match rate" in out
    assert "Causal effect" in out and "ipw ate" in out
    assert "consensus" in out


def test_report_stars_follow_the_adjusted_p():
    out = morie.mrm_report(effect=_Effect())
    line_a = [l for l in out.splitlines() if "ipw ate" in l][0]
    line_b = [l for l in out.splitlines() if "aipw" in l][0]
    assert line_a.rstrip().endswith("***")     # p.adj = 0.0004
    assert line_b.rstrip().endswith("*")       # p.adj = 0.02
    assert morie.mrm_report(effect=_Effect(), stars=False).count("*") == 0


def test_report_needs_results_on_the_effect():
    class Bad:
        pass
    with pytest.raises(ValueError, match="results"):
        morie.mrm_report(effect=Bad())
