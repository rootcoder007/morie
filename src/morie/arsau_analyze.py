# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.arsau_analyze — per-record-type ARSAU analysis pipelines.

Each public callable in this module loads one ARSAU dataset via
:func:`morie.arsau_datasets.arsau_load_*` and chains the
jurisdiction-agnostic MRM Use-of-Force primitives from
:mod:`morie.mrm_uof` over it, producing a single
:class:`morie.fn._richresult.RichResult` that holds the loaded data,
all sub-analyses, and a multi-paragraph natural-language summary.

The analyzers do not invent new statistical methods — they wire up the
generic callables against the column names that the Ontario open-data
release actually publishes. If the upstream schema changes, the
generic callables in :mod:`morie.mrm_uof` will still work; the only
patch needed will be the column-name constants below.

Public callables
----------------

- :func:`arsau_analyze_main_records` (per year, 2023 / 2024)
- :func:`arsau_analyze_individual_records` (per year, 2023 / 2024)
- :func:`arsau_analyze_probe_cycle_records` (per year, 2023 / 2024)
- :func:`arsau_analyze_weapon_records` (per year, 2023 / 2024;
  2023 requires ``allow_invalid=True``)
- :func:`arsau_analyze_aggregate_summary` (2020-2022)
- :func:`arsau_analyze_detailed_dataset` (2020-2022)

Every analyzer returns a RichResult whose ``payload`` contains the
constituent sub-results under named keys, so callers can drill into
specific tests without re-running the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from morie.arsau_datasets import (
    arsau_load_aggregate_summary,
    arsau_load_detailed_dataset,
    arsau_load_individual_records,
    arsau_load_main_records,
    arsau_load_probe_cycle_records,
    arsau_load_weapon_records,
)
from morie.fn._richresult import RichResult
from morie.mrm_uof import (
    mrm_uof_data_quality_audit,
    mrm_uof_demographic_disparity,
    mrm_uof_force_concentration,
    mrm_uof_weapon_diversity,
    mrm_uof_yoy_change,
)

# Column-name constants for the ARSAU schemas.
_MAIN_FORCE_COL = "PoliceService"
_MAIN_FORCE_TYPE_COL = "PoliceServiceType"
_MAIN_REGION_COL = "OPP_PoliceService_Region"
_MAIN_INCIDENT_TYPE_COL = "IncidentType"
_MAIN_LOCATION_PREFIX = "LocationType_"

_INDIV_RACE_COL = "Race"
_INDIV_GENDER_COL = "Gender"
_INDIV_AGE_COL = "AgeCategory"
_INDIV_OUTCOME_COL = "IndivInjuries_PhysicalInjuries"
_INDIV_KEY_COLS = ("BatchFileName", "Indiv_Index")

_WEAPON_WEAPON_COL = "Weapon"
_WEAPON_LOCATION_COL = "Location"

_PROBE_CYCLE_COL = "CEW_CartridgeProbe_CartridgeProbeCycles_Cyc"

_AGG_SECTION_COL = "SECTION"
_AGG_CATEGORY_COL = "CATEGORY"
_AGG_UNITS_COL = "UNITS OF MEASURE"
_AGG_YEAR_PREFIX = "YEAR_"


def _wrap(
    *,
    title: str,
    call: str,
    sub_results: dict[str, RichResult],
    data: pd.DataFrame,
    sidecar: dict | None,
    year_or_range: str,
    kind: str,
    language: str,
    is_valid: bool,
    extra_interpretation: str = "",
) -> RichResult:
    """Build a wrap-all RichResult from individual sub-results."""
    warnings: list[str] = []
    if not is_valid:
        warnings.append(
            "Source dataset flagged invalid by the publishing ministry — "
            "results below are presented for data-quality review only."
        )

    # Hoist warnings from sub-results so users see them at the top.
    for name, sub in sub_results.items():
        for w in sub.warnings:
            warnings.append(f"[{name}] {w}")

    summary_lines: list[tuple[str, Any]] = [
        ("Year/range", year_or_range),
        ("Kind", kind),
        ("Rows analysed", len(data)),
        ("Columns analysed", len(data.columns)),
        ("Valid", "yes" if is_valid else "no"),
        ("Sub-analyses", len(sub_results)),
    ]

    sections: list[dict[str, Any]] = []
    for name, sub in sub_results.items():
        sections.append(
            {
                "title": f"-- {name} -- ({sub.title})",
                "text": (sub.interpretation if isinstance(sub.interpretation, str) else str(sub.interpretation)),
            }
        )

    base_interp = (
        f"Ran {len(sub_results)} sub-analysis(es) over the ARSAU "
        f"{kind!r} dataset for {year_or_range!r}: "
        f"{', '.join(sub_results.keys())}. Each sub-result is "
        f"available as `result.<name>` and the underlying DataFrame "
        f"as `result.data`."
    )

    interpretation = (base_interp + " " + extra_interpretation).strip()

    return RichResult(
        title=title,
        call=call,
        summary_lines=summary_lines,
        sections=sections,
        warnings=warnings,
        interpretation=interpretation,
        payload={
            "data": data,
            "sidecar": sidecar,
            "year_or_range": year_or_range,
            "kind": kind,
            "language": language,
            "is_valid": is_valid,
            "n_rows": len(data),
            "n_cols": len(data.columns),
            "value": len(sub_results),
            **sub_results,
        },
    )


# ── 1. main_records analysis ────────────────────────────────────────


def arsau_analyze_main_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """End-to-end analysis of the ARSAU main_records CSV for one year.

    Chains:

    - :func:`morie.mrm_uof.mrm_uof_force_concentration` over
      ``PoliceService``
    - :func:`morie.mrm_uof.mrm_uof_weapon_diversity` over
      ``IncidentType x PoliceService``
    - :func:`morie.mrm_uof.mrm_uof_data_quality_audit` against the
      published CKAN sidecar (when present)

    Region-locality is NOT meaningful for main_records — only the
    ``OPP_PoliceService_Region`` field is published, and it pairs
    one column with itself. See
    :func:`arsau_analyze_detailed_dataset` for the 2020-2022 layout
    that exposes more region columns.
    """
    loaded = arsau_load_main_records(year, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    if _MAIN_FORCE_COL in df.columns:
        sub_results["force_concentration"] = mrm_uof_force_concentration(df, force_col=_MAIN_FORCE_COL)

    if _MAIN_INCIDENT_TYPE_COL in df.columns and _MAIN_FORCE_COL in df.columns:
        sub_results["incident_type_x_force"] = mrm_uof_weapon_diversity(
            df,
            weapon_col=_MAIN_INCIDENT_TYPE_COL,
            force_col=_MAIN_FORCE_COL,
        )

    sub_results["data_quality"] = mrm_uof_data_quality_audit(
        df,
        sidecar=loaded.sidecar,
    )

    return _wrap(
        title=f"ARSAU main_records analysis ({loaded.year})",
        call=f"arsau_analyze_main_records(year={year!r})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="main_records",
        language=language,
        is_valid=loaded.is_valid,
    )


# ── 2. individual_records analysis ──────────────────────────────────


def arsau_analyze_individual_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
    bootstrap_reps: int = 0,
) -> RichResult:
    """End-to-end analysis of the ARSAU individual_records CSV for one year.

    Chains demographic-disparity tests over Race, Gender, and
    AgeCategory against the ``IndivInjuries_PhysicalInjuries`` outcome
    column, plus a data-quality audit against the sidecar.

    Parameters
    ----------
    bootstrap_reps : int, default 0
        Forwarded to
        :func:`morie.mrm_uof.mrm_uof_demographic_disparity`. Set to
        e.g. 1000 to get percentile-bootstrap CIs on the risk ratios.
    """
    loaded = arsau_load_individual_records(year, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    # Tolerate the trailing-space typo in some Ontario releases.
    outcome_col_actual = None
    for c in df.columns:
        if str(c).strip().lower() == _INDIV_OUTCOME_COL.lower():
            outcome_col_actual = c
            break

    if outcome_col_actual is None:
        # Outcome column missing — still do DQ audit but skip disparity.
        sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)
        return _wrap(
            title=f"ARSAU individual_records analysis ({loaded.year})",
            call=f"arsau_analyze_individual_records(year={year!r})",
            sub_results=sub_results,
            data=df,
            sidecar=loaded.sidecar,
            year_or_range=loaded.year,
            kind="individual_records",
            language=language,
            is_valid=loaded.is_valid,
            extra_interpretation=(
                f"Disparity analysis skipped: outcome column {_INDIV_OUTCOME_COL!r} not found in this CSV."
            ),
        )

    # Coerce outcome to 0/1.  The published CSVs use "Yes"/"No"/blank
    # for boolean fields, with mixed dtypes across releases (Arrow
    # string in 2024, object in older years).  Do the coercion
    # robustly: stringify, strip, isin-filter, then map.
    outcome_str = df[outcome_col_actual].astype(str).str.strip().str.lower()
    coerce_map = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
    mask = outcome_str.isin(coerce_map.keys())
    work = df[mask].copy()
    work["_outcome"] = outcome_str[mask].map(coerce_map).astype(int)

    for demo, name in (
        (_INDIV_RACE_COL, "disparity_by_race"),
        (_INDIV_GENDER_COL, "disparity_by_gender"),
        (_INDIV_AGE_COL, "disparity_by_age"),
    ):
        if demo in work.columns:
            sub_results[name] = mrm_uof_demographic_disparity(
                work,
                demo_col=demo,
                outcome_col="_outcome",
                bootstrap_reps=bootstrap_reps,
            )

    sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)

    return _wrap(
        title=f"ARSAU individual_records analysis ({loaded.year})",
        call=f"arsau_analyze_individual_records(year={year!r})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="individual_records",
        language=language,
        is_valid=loaded.is_valid,
        extra_interpretation=(
            f"Outcome variable is {_INDIV_OUTCOME_COL!r} (coerced from "
            f"Yes/No strings to 1/0). Disparity tests use the "
            f"largest-N demographic group as the baseline; pass "
            f"bootstrap_reps&gt;0 to attach percentile CIs to the risk "
            f"ratios."
        ),
    )


# ── 3. probe_cycle_records analysis ─────────────────────────────────


def arsau_analyze_probe_cycle_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Analysis of ARSAU probe_cycle_records (CEW telemetry).

    The probe-cycle file is intentionally narrow (BatchFileName +
    Indiv_Index + a comma-separated cycle string). We compute the
    cycle-count distribution per incident and run a data-quality audit.
    """
    loaded = arsau_load_probe_cycle_records(year, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    if _PROBE_CYCLE_COL in df.columns:
        # The cycle column is a "Cyc1, Cyc2, ..." style list; count the
        # number of cycles per row.
        cycle_counts = (
            df[_PROBE_CYCLE_COL]
            .fillna("")
            .astype(str)
            .map(lambda s: 0 if not s.strip() else len([t for t in s.split(",") if t.strip()]))
        )
        descriptive = {
            "n_rows": int(len(df)),
            "n_with_cycles": int((cycle_counts > 0).sum()),
            "mean_cycles": float(cycle_counts.mean()) if len(df) else float("nan"),
            "median_cycles": float(cycle_counts.median()) if len(df) else float("nan"),
            "max_cycles": int(cycle_counts.max()) if len(df) else 0,
        }
        sub_results["cycle_distribution"] = RichResult(
            title="CEW cycle-count distribution",
            call=f"(internal) cycle parse of {_PROBE_CYCLE_COL!r}",
            summary_lines=[(k, v) for k, v in descriptive.items()],
            interpretation=(
                f"Across {descriptive['n_rows']} probe-cycle row(s), "
                f"the mean number of cycles per row is "
                f"{descriptive['mean_cycles']:.2f} (median "
                f"{descriptive['median_cycles']:.2f}, max "
                f"{descriptive['max_cycles']})."
            ),
            payload=descriptive,
        )

    sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)

    return _wrap(
        title=f"ARSAU probe_cycle_records analysis ({loaded.year})",
        call=f"arsau_analyze_probe_cycle_records(year={year!r})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="probe_cycle_records",
        language=language,
        is_valid=loaded.is_valid,
    )


# ── 4. weapon_records analysis ──────────────────────────────────────


def arsau_analyze_weapon_records(
    year: str | int,
    *,
    allow_invalid: bool = False,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Analysis of ARSAU weapon_records.

    Chains :func:`morie.mrm_uof.mrm_uof_weapon_diversity` over
    ``Weapon x Location`` (the only two categorical columns the file
    publishes) plus a data-quality audit.

    The 2023 file is the ministry-flagged-invalid release and requires
    ``allow_invalid=True``.
    """
    loaded = arsau_load_weapon_records(year, allow_invalid=allow_invalid, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    if _WEAPON_WEAPON_COL in df.columns and _WEAPON_LOCATION_COL in df.columns:
        sub_results["weapon_x_location"] = mrm_uof_weapon_diversity(
            df,
            weapon_col=_WEAPON_WEAPON_COL,
            force_col=_WEAPON_LOCATION_COL,
        )

    if _WEAPON_WEAPON_COL in df.columns:
        # Frequency table for Weapon (no force column in this file).
        wc = df[_WEAPON_WEAPON_COL].value_counts()
        rows = [[str(w), int(n), float(n) / int(wc.sum())] for w, n in wc.items()]
        sub_results["weapon_frequencies"] = RichResult(
            title="Weapon frequency distribution",
            call=f"(internal) value_counts on {_WEAPON_WEAPON_COL!r}",
            summary_lines=[
                ("Distinct weapons", int(wc.size)),
                ("Total weapon rows", int(wc.sum())),
                ("Top weapon", str(wc.index[0]) if wc.size else "-"),
                ("Top weapon share", float(wc.iloc[0]) / int(wc.sum()) if wc.size else 0.0),
            ],
            tables=[
                {
                    "title": "Weapons by frequency",
                    "headers": ["weapon", "n", "share"],
                    "rows": rows,
                }
            ],
            interpretation=(f"{int(wc.size)} distinct weapon type(s) recorded across {int(wc.sum())} weapon-row(s)."),
            payload={
                "n_distinct": int(wc.size),
                "n_total": int(wc.sum()),
                "value": int(wc.size),
            },
        )

    sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)

    extra = ""
    if not loaded.is_valid:
        extra = (
            "These results are computed for data-quality review only — "
            "the underlying file is the ministry-flagged invalid 2023 "
            "release. Do NOT use the weapon frequencies or the chi-square "
            "association in comparative analysis."
        )

    return _wrap(
        title=f"ARSAU weapon_records analysis ({loaded.year})",
        call=f"arsau_analyze_weapon_records(year={year!r}, allow_invalid={allow_invalid})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="weapon_records",
        language=language,
        is_valid=loaded.is_valid,
        extra_interpretation=extra,
    )


# ── 5. aggregate_summary analysis ───────────────────────────────────


def arsau_analyze_aggregate_summary(
    year_range: str = "2020-2022",
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Analysis of the ARSAU aggregate-summary-by-year file.

    The aggregate file is a long-format YEAR_2020 / YEAR_2021 / YEAR_2022
    panel keyed by (SECTION, CATEGORY, UNITS OF MEASURE). We rebuild
    the implied time series, run year-on-year change against the
    REPORT_SCOPE rows (the headline volume series), and surface a
    data-quality audit.
    """
    loaded = arsau_load_aggregate_summary(year_range, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    year_cols = [c for c in df.columns if c.startswith(_AGG_YEAR_PREFIX)]
    if year_cols:
        # Synthesize per-year "DataFrame" objects so the YoY callable
        # sees the same shape it does on the per-incident files.
        years = sorted(int(c.replace(_AGG_YEAR_PREFIX, "")) for c in year_cols)
        # Use the REPORT_SCOPE row "1 to 3 Subjects - Individual Reports"
        # as the headline volume series.
        mask = (
            (df[_AGG_SECTION_COL] == "REPORT_SCOPE") if _AGG_SECTION_COL in df.columns else pd.Series([True] * len(df))
        )
        headline = df[mask].iloc[0] if mask.any() else df.iloc[0]
        dfs_by_year = {}
        for y in years:
            col = f"{_AGG_YEAR_PREFIX}{y}"
            value = headline[col] if col in headline.index else 0
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            # Make a count-rows-shaped frame whose length == headline count.
            dfs_by_year[y] = pd.DataFrame({"row": range(count)}) if count > 0 else pd.DataFrame()

        sub_results["yoy_change_headline"] = mrm_uof_yoy_change(dfs_by_year=dfs_by_year)

    sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)

    return _wrap(
        title=f"ARSAU aggregate_summary analysis ({loaded.year})",
        call=f"arsau_analyze_aggregate_summary(year_range={year_range!r})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="aggregate_summary",
        language=language,
        is_valid=loaded.is_valid,
        extra_interpretation=(
            "Year-on-year change is computed against the "
            "'1 to 3 Subjects - Individual Reports' REPORT_SCOPE row, "
            "which is the headline volume metric in the ministry's "
            "annual technical reports."
        ),
    )


# ── 6. detailed_dataset analysis ────────────────────────────────────


def arsau_analyze_detailed_dataset(
    year_range: str = "2020-2022",
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Wide-format analysis of the 2020-2022 detailed-incident dataset.

    Chains:
      - force_concentration on POLICE_SERVICE
      - weapon_diversity on (POLICE_SERVICE x ASSIGNMENT_TYPE)
      - yoy_change on REPORTING_YEAR
      - data_quality audit
    """
    loaded = arsau_load_detailed_dataset(year_range, language=language, data_dir=data_dir)
    df: pd.DataFrame = loaded.data

    sub_results: dict[str, RichResult] = {}

    force_col = "POLICE_SERVICE" if "POLICE_SERVICE" in df.columns else None
    year_col = "REPORTING_YEAR" if "REPORTING_YEAR" in df.columns else None
    assignment_col = "ASSIGNMENT_TYPE" if "ASSIGNMENT_TYPE" in df.columns else None

    if force_col is not None:
        sub_results["force_concentration"] = mrm_uof_force_concentration(df, force_col=force_col)
    if force_col is not None and assignment_col is not None:
        sub_results["assignment_x_force"] = mrm_uof_weapon_diversity(df, weapon_col=assignment_col, force_col=force_col)
    if year_col is not None:
        sub_results["yoy_change"] = mrm_uof_yoy_change(df=df, year_col=year_col)

    sub_results["data_quality"] = mrm_uof_data_quality_audit(df, sidecar=loaded.sidecar)

    return _wrap(
        title=f"ARSAU detailed_dataset analysis ({loaded.year})",
        call=f"arsau_analyze_detailed_dataset(year_range={year_range!r})",
        sub_results=sub_results,
        data=df,
        sidecar=loaded.sidecar,
        year_or_range=loaded.year,
        kind="detailed_dataset",
        language=language,
        is_valid=loaded.is_valid,
    )


__all__ = [
    "arsau_analyze_main_records",
    "arsau_analyze_individual_records",
    "arsau_analyze_probe_cycle_records",
    "arsau_analyze_weapon_records",
    "arsau_analyze_aggregate_summary",
    "arsau_analyze_detailed_dataset",
]
