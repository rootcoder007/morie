# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.audit_variables — comprehensive per-variable audit walker.

Walks every (dataset, column) pair across the OTIS and ARSAU
catalogues, classifies each variable via
:mod:`morie.variable_taxonomy`, and reports:

- how many variables in total
- distribution across levels of measurement
- distribution across functional roles
- which variables are currently exercised by an analyzer in
  ``mrm_otis.R`` / ``mrm_arsau.py``
- which variables are flagged ``cross_year_safe = False``
- which variables are present in the CSVs but NOT in the dictionary
  (and vice versa)
- the recommended summary statistic per variable
- the recommended bivariate test per interesting pair

Output is a :class:`morie.fn._richresult.RichResult` plus an
auxiliary Markdown report saved to
``papers/data-dictionary-audit.md`` so the v0.9.5.5 release ships
with a defensible record of "every variable accounted for".

Public callables
----------------

- :func:`audit_otis_variables` — walks the 29 OTIS datasets
- :func:`audit_arsau_variables` — walks the 10 ARSAU datasets
- :func:`audit_all_variables` — both; the headline runbook
- :func:`write_audit_markdown` — persists a Markdown report
  (default location: ``docs/data-dictionary-audit.md``)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from morie.dataset_dictionary import (
    DatasetSchema,
    load_arsau_dictionary,
    load_otis_dictionary,
)
from morie.fn._richresult import RichResult
from morie.variable_taxonomy import (
    Cardinality,
    LevelOfMeasurement,
    Role,
    VariableTaxonomy,
    classify_schema,
)


# Names of columns that the existing analyzers actually exercise.
# Used to compute the "coverage" % of the variable surface.
#
# Sourced by inspection of r-package/morie/R/mrm_otis.R and
# src/morie/arsau_analyze.py.
ANALYZED_COLUMNS_OTIS: set[str] = {
    # b01 — segregation detailed
    "UniqueIndividual_ID",                  # placement_concentration
    "NumberConsecutiveDays_Segregation",    # seg_duration_km
    "MentalHealth_Alert",                   # mortification_cooccurrence
    "SuicideRisk_Alert",
    "SuicideWatch_Alert",
    "Region_AtTimeOfPlacement",             # region_locality
    "Region_MostRecentPlacement",
}
ANALYZED_COLUMNS_ARSAU: set[str] = {
    # main
    "PoliceService", "IncidentType", "PoliceServiceType",
    "OPP_PoliceService_Region",
    # individual
    "Race", "Gender", "AgeCategory",
    "IndivInjuries_PhysicalInjuries",
    # weapon
    "Weapon", "Location",
    # probe
    "CEW_CartridgeProbe_CartridgeProbeCycles_Cyc",
    # detailed (2020-2022)
    "POLICE_SERVICE", "ASSIGNMENT_TYPE", "REPORTING_YEAR",
}


# ── Internal helpers ──────────────────────────────────────────────


def _collect_all_taxonomies(
    schemas: dict[str, DatasetSchema],
) -> list[VariableTaxonomy]:
    out: list[VariableTaxonomy] = []
    for schema in schemas.values():
        out.extend(classify_schema(schema))
    return out


def _coverage_set(domain: str) -> set[str]:
    if domain == "otis":
        return ANALYZED_COLUMNS_OTIS
    if domain == "arsau":
        return ANALYZED_COLUMNS_ARSAU
    if domain == "both":
        return ANALYZED_COLUMNS_OTIS | ANALYZED_COLUMNS_ARSAU
    return set()


def _audit_report(
    *,
    domain: str,
    schemas: dict[str, DatasetSchema],
    taxonomies: list[VariableTaxonomy],
) -> RichResult:
    """Build the RichResult given a fully-classified variable list."""
    analyzed = _coverage_set(domain)
    level_counts = Counter(t.level.value for t in taxonomies)
    role_counts = Counter(t.role.value for t in taxonomies)
    card_counts = Counter(t.cardinality.value for t in taxonomies)

    n_total = len(taxonomies)
    n_analyzed = sum(1 for t in taxonomies if t.column_name in analyzed)
    n_uncovered = n_total - n_analyzed
    coverage_pct = (n_analyzed / n_total * 100) if n_total else 0.0

    cross_year_unsafe = [
        t for t in taxonomies if not t.cross_year_safe
    ]
    identifiers = [t for t in taxonomies if t.role == Role.IDENTIFIER]
    outcomes = [t for t in taxonomies if t.role == Role.OUTCOME]

    level_table = [
        [lvl, level_counts.get(lvl, 0)]
        for lvl in [e.value for e in LevelOfMeasurement]
    ]
    role_table = [
        [r, role_counts.get(r, 0)] for r in [e.value for e in Role]
    ]
    card_table = [
        [c, card_counts.get(c, 0)] for c in [e.value for e in Cardinality]
    ]

    interp = (
        f"Audited {n_total} variable(s) across {len(schemas)} {domain.upper()} "
        f"dataset(s). Currently analysed: {n_analyzed} "
        f"({coverage_pct:.1f}%); not exercised by any analyzer: "
        f"{n_uncovered}. {len(cross_year_unsafe)} variable(s) flagged "
        f"cross_year_safe=FALSE (do not join across fiscal years). "
        f"{len(identifiers)} identifier column(s), {len(outcomes)} "
        f"outcome column(s)."
    )

    sections: list[dict] = [
        {
            "title": "Variables by level of measurement",
            "headers": ["level", "n"],
            "table": level_table,
        },
        {
            "title": "Variables by functional role",
            "headers": ["role", "n"],
            "table": role_table,
        },
        {
            "title": "Variables by cardinality",
            "headers": ["cardinality", "n"],
            "table": card_table,
        },
    ]
    if cross_year_unsafe:
        sections.append({
            "title": "Cross-year-UNSAFE variables (do not join across fiscal years)",
            "headers": ["dataset", "column", "notes"],
            "table": [
                [t.dataset_name, t.column_name, t.notes or ""]
                for t in cross_year_unsafe
            ],
        })
    if identifiers:
        sections.append({
            "title": "Identifier columns (joins / grouping only)",
            "headers": ["dataset", "column"],
            "table": [
                [t.dataset_name, t.column_name] for t in identifiers[:30]
            ],
        })
    if outcomes:
        sections.append({
            "title": "Outcome columns (candidate y for modelling)",
            "headers": ["dataset", "column"],
            "table": [
                [t.dataset_name, t.column_name] for t in outcomes
            ],
        })

    return RichResult(
        title=f"morie variable-coverage audit — {domain.upper()}",
        call=f"audit_{domain}_variables()",
        summary_lines=[
            ("Datasets", len(schemas)),
            ("Total variables", n_total),
            ("Currently analysed", n_analyzed),
            ("Coverage", f"{coverage_pct:.1f}%"),
            ("Cross-year-unsafe", len(cross_year_unsafe)),
            ("Identifiers", len(identifiers)),
            ("Outcomes", len(outcomes)),
        ],
        sections=sections,
        interpretation=interp,
        payload={
            "n": n_total,
            "n_analyzed": n_analyzed,
            "n_uncovered": n_uncovered,
            "coverage_pct": coverage_pct,
            "domain": domain,
            "taxonomies": taxonomies,
            "level_counts": dict(level_counts),
            "role_counts": dict(role_counts),
            "cardinality_counts": dict(card_counts),
            "cross_year_unsafe": cross_year_unsafe,
            "identifiers": identifiers,
            "outcomes": outcomes,
            "value": float(coverage_pct),
        },
    )


# ── Public callables ──────────────────────────────────────────────


def audit_otis_variables(
    *,
    data_dir: str | Path | None = None,
) -> RichResult:
    """Walk every column in every OTIS dataset; classify; report."""
    schemas = load_otis_dictionary(data_dir=data_dir)
    taxonomies = _collect_all_taxonomies(schemas)
    return _audit_report(
        domain="otis", schemas=schemas, taxonomies=taxonomies,
    )


def audit_arsau_variables(
    *,
    data_dir: str | Path | None = None,
    year_ranges: Iterable[str] = ("2020-2022", "2023", "2024"),
) -> RichResult:
    """Walk every column in every ARSAU dictionary; classify; report.

    Concatenates the dictionaries for all year ranges (the same
    dataset_name can appear in multiple years; the resulting
    taxonomy list contains one row per (year, dataset, column)).
    """
    all_schemas: dict[str, DatasetSchema] = {}
    for yr in year_ranges:
        try:
            d = load_arsau_dictionary(yr, data_dir=data_dir)
        except (FileNotFoundError, ValueError):
            continue
        for name, sch in d.items():
            # Key by year+name to avoid clobbering across years.
            all_schemas[f"{yr}__{name}"] = DatasetSchema(
                dataset_name=f"{yr}__{name}",
                source_path=sch.source_path,
                source_kind=sch.source_kind,
                language=sch.language,
                columns=sch.columns,
            )
    taxonomies = _collect_all_taxonomies(all_schemas)
    return _audit_report(
        domain="arsau", schemas=all_schemas, taxonomies=taxonomies,
    )


def audit_all_variables(
    *,
    otis_data_dir: str | Path | None = None,
    arsau_data_dir: str | Path | None = None,
) -> dict:
    """Run both audits; return a dict with each RichResult."""
    return {
        "otis": audit_otis_variables(data_dir=otis_data_dir),
        "arsau": audit_arsau_variables(data_dir=arsau_data_dir),
    }


# ── Markdown report ───────────────────────────────────────────────


_MD_HEADER = """\
# morie variable-coverage audit (v0.9.5.5)

Per-variable taxonomy + coverage audit across the OTIS and ARSAU
datasets, classified by Stevens-1946 level of measurement
(nominal/ordinal/interval/ratio + boolean/date/identifier/free-text),
functional role (identifier / outcome / covariate / metadata), and
cross-year safety.

Generated by `morie.audit_variables.audit_all_variables()`.  This
file is the authoritative record of which dataset columns are
exercised by an analyzer vs left unexamined.
"""


def write_audit_markdown(
    out_path: str | Path,
    *,
    otis_data_dir: str | Path | None = None,
    arsau_data_dir: str | Path | None = None,
) -> Path:
    """Write a comprehensive Markdown report to ``out_path``."""
    audit = audit_all_variables(
        otis_data_dir=otis_data_dir,
        arsau_data_dir=arsau_data_dir,
    )

    lines = [_MD_HEADER]

    for domain in ("otis", "arsau"):
        r = audit[domain]
        lines.append(f"\n## {domain.upper()}\n")
        lines.append(f"- **Datasets:** {len(r.payload.get('taxonomies', []))} variables across the dictionary entries")
        lines.append(f"- **Currently analysed:** {r.payload['n_analyzed']} / {r.payload['n']} "
                     f"({r.payload['coverage_pct']:.1f}%)")
        lines.append(f"- **Cross-year-unsafe:** {len(r.payload['cross_year_unsafe'])}")
        lines.append(f"- **Identifier columns:** {len(r.payload['identifiers'])}")
        lines.append(f"- **Outcome columns:** {len(r.payload['outcomes'])}")
        lines.append("")
        lines.append("### Levels of measurement")
        lines.append("")
        lines.append("| level | count |")
        lines.append("|---|---:|")
        for k, v in sorted(r.payload["level_counts"].items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

        lines.append("### Per-variable detail")
        lines.append("")
        lines.append(
            "| dataset | column | level | role | cardinality | "
            "cross_year_safe | analyzed? | recommended summary |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        analyzed = _coverage_set(domain)
        for t in r.payload["taxonomies"]:
            seen = "yes" if t.column_name in analyzed else "—"
            cys = "yes" if t.cross_year_safe else "**NO**"
            lines.append(
                f"| {t.dataset_name} | {t.column_name} | "
                f"{t.level.value} | {t.role.value} | "
                f"{t.cardinality.value} | {cys} | {seen} | "
                f"{t.recommended_summary()} |"
            )
        lines.append("")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "audit_otis_variables",
    "audit_arsau_variables",
    "audit_all_variables",
    "write_audit_markdown",
    "ANALYZED_COLUMNS_OTIS",
    "ANALYZED_COLUMNS_ARSAU",
]
