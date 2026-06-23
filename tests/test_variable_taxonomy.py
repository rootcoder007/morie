# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for src/morie/variable_taxonomy.py — every classification
heuristic + every invariant override is pinned.

The audit numbers in papers/data-dictionary-audit.md are sensitive
to taxonomy changes; these tests guard against silent regressions
in the level-of-measurement / role / cross-year-safety inferences.
"""

from __future__ import annotations

import pytest

from morie.dataset_dictionary import ColumnSpec
from morie.variable_taxonomy import (
    INVARIANT_OVERRIDES,
    Cardinality,
    LevelOfMeasurement,
    Role,
    VariableTaxonomy,
    classify_variable,
)

# ── Identifier detection ───────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "_id",
        "_Id",
        "Id_",
        "RecordID",
        "UniqueIndividual_ID",
        "BatchFileName",
        "Indiv_Index",
    ],
)
def test_identifier_names_classified_as_identifier(name):
    spec = ColumnSpec(name=name, dtype="string", description_en="")
    t = classify_variable(spec, dataset_name="any")
    assert t.level == LevelOfMeasurement.IDENTIFIER
    assert t.role == Role.IDENTIFIER


# ── Boolean detection ──────────────────────────────────────────────


def test_dtype_bool_yields_boolean():
    spec = ColumnSpec(name="MentalHealth_Alert", dtype="bool")
    t = classify_variable(spec, dataset_name="b01")
    assert t.level == LevelOfMeasurement.BOOLEAN
    assert t.cardinality == Cardinality.BINARY


def test_yesno_valid_values_yields_boolean():
    spec = ColumnSpec(
        name="IndivInjuries_PhysicalInjuries",
        dtype="string",
        valid_values=("Yes", "No"),
    )
    t = classify_variable(spec, dataset_name="uof_individual_records")
    assert t.level == LevelOfMeasurement.BOOLEAN


# ── Ordinal vs nominal ─────────────────────────────────────────────


def test_age_category_is_ordinal():
    spec = ColumnSpec(
        name="Age_Category",
        dtype="string",
        valid_values=("18 to 24", "25 to 49", "50+"),
    )
    t = classify_variable(spec, dataset_name="b01")
    assert t.level == LevelOfMeasurement.ORDINAL


def test_race_without_order_is_nominal():
    spec = ColumnSpec(
        name="Race",
        dtype="string",
        valid_values=("White", "Black", "Asian", "Indigenous"),
    )
    t = classify_variable(spec, dataset_name="uof_individual_records")
    assert t.level == LevelOfMeasurement.NOMINAL


# ── Ratio (counts) ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "NumberConsecutiveDays_Segregation",
        "Number_Of_Placements",
        "NumberTeam",
        "NUMBER_POLICE_OFFICERS_INVOLVED",
    ],
)
def test_count_like_int_names_classified_as_ratio(name):
    spec = ColumnSpec(name=name, dtype="int", description_en="")
    t = classify_variable(spec, dataset_name="any")
    assert t.level == LevelOfMeasurement.RATIO


def test_float_dtype_yields_ratio():
    spec = ColumnSpec(name="some_continuous_score", dtype="float")
    t = classify_variable(spec, dataset_name="any")
    assert t.level == LevelOfMeasurement.RATIO


# ── Cross-year-safety overrides (THE CRITICAL INVARIANTS) ──────────


def test_otis_unique_individual_id_is_cross_year_unsafe():
    """The OTIS data dictionary itself states the ID is randomly
    reassigned every fiscal year.  Any cross-year join on this column
    is statistically invalid.  This test pins the invariant so a
    silent regression in the override registry fails the build.
    """
    spec = ColumnSpec(
        name="UniqueIndividual_ID",
        dtype="string",
        description_en="A random number assigned to an individual",
    )
    for dataset_id in ("b01", "a01"):
        t = classify_variable(spec, dataset_name=dataset_id)
        assert t.cross_year_safe is False, (
            f"OTIS {dataset_id}/UniqueIndividual_ID must be cross_year_safe=False — see data dictionary."
        )
        assert t.role == Role.IDENTIFIER
        # The notes may say "fiscal year" or "fiscal-year"; either is fine.
        notes_lc = (t.notes or "").lower()
        assert "fiscal year" in notes_lc or "fiscal-year" in notes_lc


def test_arsau_batchfilename_is_identifier():
    spec = ColumnSpec(name="BatchFileName", dtype="string")
    for dataset_id in (
        "uof_main_records",
        "uof_individual_records",
        "uof_weapon_records",
        "uof_probe_cycle_records",
    ):
        t = classify_variable(spec, dataset_name=dataset_id)
        assert t.role == Role.IDENTIFIER


def test_arsau_outcome_column_is_marked_outcome():
    spec = ColumnSpec(
        name="IndivInjuries_PhysicalInjuries",
        dtype="bool",
        valid_values=("Yes", "No"),
    )
    t = classify_variable(spec, dataset_name="uof_individual_records")
    assert t.role == Role.OUTCOME


# ── Override registry integrity ────────────────────────────────────


def test_invariant_overrides_registry_is_not_empty():
    assert len(INVARIANT_OVERRIDES) > 0


def test_overrides_have_valid_field_names():
    valid_fields = {
        "level",
        "cardinality",
        "role",
        "cross_year_safe",
        "dictionary_described",
        "valid_values",
        "nullable",
        "raw_dtype",
        "notes",
        "source",
    }
    for (ds, col), patch in INVARIANT_OVERRIDES.items():
        unknown = set(patch.keys()) - valid_fields
        assert not unknown, f"Override for ({ds}, {col}) has unknown fields: {unknown}"


# ── Recommended summary string per level ───────────────────────────


@pytest.mark.parametrize(
    "level,fragment",
    [
        (LevelOfMeasurement.BOOLEAN, "Wilson"),
        (LevelOfMeasurement.NOMINAL, "chi-square"),
        (LevelOfMeasurement.ORDINAL, "median"),
        (LevelOfMeasurement.RATIO, "Pareto"),
        (LevelOfMeasurement.INTERVAL, "regression"),
        (LevelOfMeasurement.IDENTIFIER, "identifier"),
    ],
)
def test_recommended_summary_mentions_appropriate_method(level, fragment):
    spec = ColumnSpec(name="X", dtype="string")
    t = VariableTaxonomy(
        dataset_name="X",
        column_name="X",
        level=level,
        cardinality=Cardinality.UNKNOWN,
        role=Role.COVARIATE,
    )
    assert fragment.lower() in t.recommended_summary().lower()


# ── Recommended pair test (the dispatcher logic) ───────────────────


def _tax(level: LevelOfMeasurement) -> VariableTaxonomy:
    return VariableTaxonomy(
        dataset_name="X",
        column_name="X",
        level=level,
        cardinality=Cardinality.UNKNOWN,
        role=Role.COVARIATE,
    )


def test_nominal_vs_nominal_yields_chi_square():
    a = _tax(LevelOfMeasurement.NOMINAL)
    b = _tax(LevelOfMeasurement.NOMINAL)
    assert "chi-square" in a.recommended_pair_test(b).lower()


def test_ordinal_vs_ordinal_yields_spearman():
    a = _tax(LevelOfMeasurement.ORDINAL)
    b = _tax(LevelOfMeasurement.ORDINAL)
    assert "spearman" in a.recommended_pair_test(b).lower()


def test_interval_vs_ratio_yields_pearson_or_spearman():
    a = _tax(LevelOfMeasurement.INTERVAL)
    b = _tax(LevelOfMeasurement.RATIO)
    msg = a.recommended_pair_test(b).lower()
    assert "pearson" in msg or "spearman" in msg


def test_nominal_vs_ratio_yields_anova_or_kruskal():
    a = _tax(LevelOfMeasurement.NOMINAL)
    b = _tax(LevelOfMeasurement.RATIO)
    msg = a.recommended_pair_test(b).lower()
    assert "anova" in msg or "kruskal" in msg


def test_identifier_pair_refuses_test():
    a = _tax(LevelOfMeasurement.IDENTIFIER)
    b = _tax(LevelOfMeasurement.RATIO)
    msg = a.recommended_pair_test(b).lower()
    assert "identifier" in msg


# ── End-to-end audit smoke (lightweight; full audit is in
#    audit_variables tests) ───────────────────────────────────────


def test_audit_returns_real_numbers():
    """The audit walker should classify >100 OTIS variables and
    >100 ARSAU variables given the real dictionaries on disk."""
    import os

    if not (os.environ.get("MORIE_OTIS_DIR") and os.environ.get("MORIE_ARSAU_DIR")):
        pytest.skip("MORIE_OTIS_DIR and MORIE_ARSAU_DIR must be set")
    from morie.audit_variables import audit_all_variables

    audit = audit_all_variables()
    assert audit["otis"].n > 100
    assert audit["arsau"].n > 100
    assert audit["otis"].coverage_pct > 0
    assert audit["arsau"].coverage_pct > 0
    # The flagship invariant: at least one cross-year-unsafe column.
    assert len(audit["otis"].cross_year_unsafe) >= 1
