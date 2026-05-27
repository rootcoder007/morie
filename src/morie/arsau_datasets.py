# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.arsau_datasets — loaders + registry for Ontario ARSAU data.

ARSAU = the Ontario Ministry of the Solicitor General's
provincial release of Police Use-of-Force incident records (formally
"Race-Based and Identity-Based Data on Police Use of Force in
Ontario"). The data are published on the Ontario Data Catalogue
(``data.ontario.ca``) as a multi-year CKAN package; the public-facing
landing page is
https://data.ontario.ca/dataset/police-use-of-force-race-based-data.

This module ships:

- :data:`ARSAU_REGISTRY` — every (year_or_range, kind) tuple the
  ministry currently publishes, with the corresponding CSV filename,
  CKAN ``resource_id`` for the sidecar, expected row / column counts
  observed on disk, and a per-entry ``is_valid`` flag. The
  2023 ``uof_weapon_records_invaliddata.csv`` is flagged
  ``is_valid=False`` because the ministry itself marked the file as
  containing invalid data (per the published technical notes); no
  loader will return its rows without the user opting in via
  ``allow_invalid=True``.
- :func:`read_sidecar` / :func:`sidecar_to_frame` / :func:`sidecar_schema`
  — generic CKAN ``datastore_search`` JSON readers, used both by the
  loaders here and by callers who want offline access to column
  metadata.
- Per-record-type loaders, returning :class:`morie.fn._richresult.RichResult`:
  :func:`arsau_load_main_records`, :func:`arsau_load_individual_records`,
  :func:`arsau_load_probe_cycle_records`, :func:`arsau_load_weapon_records`,
  :func:`arsau_load_aggregate_summary`, :func:`arsau_load_detailed_dataset`.
- Discovery callables:
  :func:`arsau_available_years`, :func:`arsau_available_datasets`,
  :func:`arsau_describe`.

Portability
-----------

This module NEVER hard-codes paths to the maintainer's workstation.
All file resolution goes through
:func:`morie._datapaths.resolve_data_dir`, which honours (in order):

1. an explicit ``data_dir=`` argument
2. the ``MORIE_ARSAU_DIR`` environment variable
3. ``MORIE_DATA_DIR/arsau``
4. the platform-appropriate user-data directory
   (``~/Library/Application Support/morie/arsau`` on macOS,
   ``~/.local/share/morie/arsau`` on Linux,
   ``%APPDATA%/morie/arsau`` on Windows) — only if it already exists
5. a bundled tiny fixture under ``morie/data/arsau/`` — present in the
   wheel for unit tests + tutorials only

Set the env var or pass ``data_dir=`` on every loader call to opt in
to the full upstream release.

2023 weapon data — explicit invalidity gate
-------------------------------------------

The 2023 release of ARSAU shipped a separate
``uof_weapon_records_invaliddata.csv`` file rather than the usual
``uof_weapon_records.csv``. The ministry's technical report for 2023
flagged the weapon-record submissions as data-quality non-compliant
(under-reporting + format drift across the participating services),
and the open-data release was renamed accordingly. We therefore:

- expose the file in :data:`ARSAU_REGISTRY` so it remains discoverable,
- mark its entry ``is_valid=False``,
- raise :class:`ValueError` from :func:`arsau_load_weapon_records`
  for year 2023 unless the caller passes ``allow_invalid=True``,
- if ``allow_invalid=True`` is set, return a RichResult whose
  ``interpretation`` opens with an explicit warning paragraph and
  whose ``warnings`` list flags the invalidity so downstream
  consumers cannot accidentally treat the rows as comparable to the
  2022 / 2024 weapon records.

Examples
--------

>>> from morie.arsau_datasets import (                        # doctest: +SKIP
...     arsau_available_years, arsau_available_datasets,
...     arsau_load_main_records,
... )
>>> arsau_available_years()                                   # doctest: +SKIP
RichResult(... years=['2020-2022', '2023', '2024'])
>>> arsau_load_main_records(2024).n_rows                      # doctest: +SKIP
11326
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from morie._datapaths import resolve_data_dir
from morie.dataset_dictionary import (
    ColumnSpec,
    DatasetSchema,
    parse_ckan_sidecar,
)
from morie.fn._richresult import RichResult

# ── Registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArsauDataset:
    """Metadata for a single (year_or_range, kind) entry in ARSAU."""

    year_or_range: str
    """e.g. ``"2020-2022"``, ``"2023"``, ``"2024"``."""

    kind: str
    """One of: ``main_records`` | ``individual_records`` |
    ``probe_cycle_records`` | ``weapon_records`` |
    ``aggregate_summary`` | ``detailed_dataset``."""

    csv_filename: str
    """Filename of the CSV under ``<arsau_root>/<year_or_range>/``."""

    sidecar_filename: Optional[str]
    """Filename of the matching CKAN sidecar JSON, or ``None`` if the
    ministry did not publish one (e.g. the 2023 weapon-records file)."""

    expected_rows: int
    """Number of data rows observed when the registry was last
    refreshed, used for smoke-test sanity checking."""

    expected_cols: int
    """Number of columns observed."""

    is_valid: bool
    """``True`` for all ARSAU files except the 2023 weapon-records,
    which the ministry flagged as containing invalid data."""

    description_en: str
    description_fr: str

    def csv_path(self, root: Path) -> Path:
        return root / self.year_or_range / self.csv_filename

    def sidecar_path(self, root: Path) -> Optional[Path]:
        if self.sidecar_filename is None:
            return None
        return root / self.year_or_range / self.sidecar_filename


# Hard-coded resource_id → CSV mapping for each (year_or_range, kind),
# verified by matching CKAN ``fields[].id`` against CSV header columns
# on disk. The ministry publishes one CSV per record type plus one
# CKAN sidecar JSON per CSV (except for the 2023 weapon records).
ARSAU_REGISTRY: dict[tuple[str, str], ArsauDataset] = {
    # ── 2020-2022 ──────────────────────────────────────────────────
    ("2020-2022", "aggregate_summary"): ArsauDataset(
        year_or_range="2020-2022",
        kind="aggregate_summary",
        csv_filename="useofforce_agrregatesummarybyyear_2020-2022.csv",
        sidecar_filename="7560d405-444c-4340-95c4-f73849015501.json",
        expected_rows=151,
        expected_cols=6,
        is_valid=True,
        description_en=(
            "Aggregate annual summary of Ontario police use-of-force "
            "reports for 2020-2022, one row per service-year. Useful "
            "as the year-level reference series before drilling into "
            "the detailed_dataset."
        ),
        description_fr=(
            "Sommaire annuel agrégé des rapports d'usage de la force "
            "policière en Ontario pour 2020-2022, une ligne par "
            "service et par année."
        ),
    ),
    ("2020-2022", "detailed_dataset"): ArsauDataset(
        year_or_range="2020-2022",
        kind="detailed_dataset",
        csv_filename="useofforce_detaileddataset_2020-2022.csv",
        sidecar_filename="2150ac23-4e55-474a-b61f-81baf6850851.json",
        expected_rows=23092,
        expected_cols=167,
        is_valid=True,
        description_en=(
            "Detailed incident-level Ontario use-of-force dataset for "
            "2020-2022 (single combined file). 167-column wide layout "
            "where each incident-attribute is a separate column, "
            "including incident type, weapon, force, attire, region, "
            "and demographic indicators."
        ),
        description_fr=(
            "Jeu de données détaillé au niveau de l'incident pour le "
            "recours à la force par la police en Ontario, 2020-2022 "
            "(fichier combiné unique). Mise en page de 167 colonnes."
        ),
    ),
    # ── 2023 ───────────────────────────────────────────────────────
    ("2023", "individual_records"): ArsauDataset(
        year_or_range="2023",
        kind="individual_records",
        csv_filename="uof_individual_records.csv",
        sidecar_filename="133c73fa-9d8e-435e-8c6d-7d1e14d1e88d.json",
        expected_rows=12805,
        expected_cols=112,
        is_valid=True,
        description_en=(
            "2023 Ontario use-of-force individual-records dataset, "
            "one row per civilian individual involved in each incident."
        ),
        description_fr=(
            "Jeu de données 2023 sur les enregistrements individuels "
            "lors du recours à la force par la police en Ontario, une "
            "ligne par civil impliqué."
        ),
    ),
    ("2023", "main_records"): ArsauDataset(
        year_or_range="2023",
        kind="main_records",
        csv_filename="uof_main_records.csv",
        sidecar_filename="94f303a2-963e-4fd1-958d-6681b310cb6d.json",
        expected_rows=10935,
        expected_cols=64,
        is_valid=True,
        description_en=(
            "2023 Ontario use-of-force main records: one row per "
            "incident with police-service, location, time, and "
            "incident-type attributes."
        ),
        description_fr=(
            "2023 Ontario, enregistrements principaux du recours à la "
            "force policière: une ligne par incident."
        ),
    ),
    ("2023", "probe_cycle_records"): ArsauDataset(
        year_or_range="2023",
        kind="probe_cycle_records",
        csv_filename="uof_probe_cycle_records.csv",
        sidecar_filename="339b9e63-9521-44a6-8719-c2cb9aa39a8a.json",
        expected_rows=1136,
        expected_cols=3,
        is_valid=True,
        description_en=(
            "2023 Ontario use-of-force probe-cycle records, one row "
            "per Conducted Energy Weapon probe-cycle (a high-resolution "
            "telemetry slice of CEW deployments)."
        ),
        description_fr=(
            "Enregistrements 2023 du cycle de sonde de l'arme à énergie "
            "dirigée, une ligne par cycle."
        ),
    ),
    ("2023", "weapon_records"): ArsauDataset(
        year_or_range="2023",
        kind="weapon_records",
        csv_filename="uof_weapon_records_invaliddata.csv",
        sidecar_filename=None,
        expected_rows=8711,
        expected_cols=4,
        is_valid=False,
        description_en=(
            "2023 Ontario use-of-force weapon records — INVALID per "
            "the ministry's technical report. The 2023 weapon-records "
            "submissions did not meet data-quality standards across "
            "participating services and the open-data release was "
            "renamed to '_invaliddata' accordingly. Do not use these "
            "rows for comparative or quantitative analysis."
        ),
        description_fr=(
            "Enregistrements 2023 sur les armes — DONNÉES INVALIDES "
            "selon le rapport technique du ministère. Ne pas utiliser "
            "pour des analyses comparatives ou quantitatives."
        ),
    ),
    # ── 2024 ───────────────────────────────────────────────────────
    ("2024", "individual_records"): ArsauDataset(
        year_or_range="2024",
        kind="individual_records",
        csv_filename="uof_individual_records.csv",
        sidecar_filename="690d4c5e-095e-49a0-bbab-b7fc680f3c6b.json",
        expected_rows=12921,
        expected_cols=112,
        is_valid=True,
        description_en=(
            "2024 Ontario use-of-force individual-records dataset, "
            "one row per civilian individual involved in each incident."
        ),
        description_fr=(
            "Jeu de données 2024 sur les enregistrements individuels "
            "lors du recours à la force, une ligne par civil impliqué."
        ),
    ),
    ("2024", "main_records"): ArsauDataset(
        year_or_range="2024",
        kind="main_records",
        csv_filename="uof_main_records.csv",
        sidecar_filename="ea9dc29c-b4f1-4426-b1f2-974ce995aca1.json",
        expected_rows=10849,
        expected_cols=65,
        is_valid=True,
        description_en=(
            "2024 Ontario use-of-force main records: one row per "
            "incident. pandas correctly parses 10849 logical records "
            "from the CSV (the higher physical line count seen by some "
            "tools reflects embedded newlines inside quoted fields)."
        ),
        description_fr=(
            "Enregistrements principaux du recours à la force 2024 en "
            "Ontario: une ligne par incident. pandas analyse "
            "correctement 10849 enregistrements logiques."
        ),
    ),
    ("2024", "probe_cycle_records"): ArsauDataset(
        year_or_range="2024",
        kind="probe_cycle_records",
        csv_filename="uof_probe_cycle_records.csv",
        sidecar_filename="76875b6a-4352-4722-a3f6-997cc220dc4f.json",
        expected_rows=972,
        expected_cols=3,
        is_valid=True,
        description_en=(
            "2024 Ontario CEW probe-cycle records, one row per cycle."
        ),
        description_fr=(
            "Enregistrements 2024 du cycle de sonde CEW, une ligne par "
            "cycle."
        ),
    ),
    ("2024", "weapon_records"): ArsauDataset(
        year_or_range="2024",
        kind="weapon_records",
        csv_filename="uof_weapon_records.csv",
        sidecar_filename="2c1ab494-d636-4c17-9699-3819112982a5.json",
        expected_rows=9282,
        expected_cols=5,
        is_valid=True,
        description_en=(
            "2024 Ontario use-of-force weapon records, one row per "
            "weapon used in each incident. Resumes the valid annual "
            "series after the 2023 invalid-data interruption."
        ),
        description_fr=(
            "Enregistrements 2024 sur les armes utilisées lors du "
            "recours à la force, une ligne par arme. Reprend la série "
            "annuelle valide après l'interruption de 2023."
        ),
    ),
}


# Allowed values per discovery callable.
ARSAU_YEARS: tuple[str, ...] = tuple(sorted({k[0] for k in ARSAU_REGISTRY}))
ARSAU_KINDS: tuple[str, ...] = tuple(sorted({k[1] for k in ARSAU_REGISTRY}))


# ── Sidecar reading ──────────────────────────────────────────────────


def read_sidecar(path: str | Path) -> dict:
    """Parse a CKAN ``datastore_search`` JSON file from disk.

    Handles both bare ``{"fields": [...], "records": [...]}`` and the
    common ``{"result": {"fields": [...], "records": [...]}}`` wrapper.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        With keys ``"fields"`` and ``"records"`` (always present;
        empty lists if the file is malformed).
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        payload = json.load(f)
    if "fields" in payload and "records" in payload:
        return {"fields": payload["fields"], "records": payload["records"]}
    if "result" in payload and isinstance(payload["result"], dict):
        result = payload["result"]
        return {
            "fields": result.get("fields", []),
            "records": result.get("records", []),
        }
    return {"fields": payload.get("fields", []), "records": payload.get("records", [])}


def sidecar_schema(sidecar: dict) -> list[dict[str, Any]]:
    """Extract a simplified ``[{name, type, notes}, ...]`` from a sidecar dict."""
    out: list[dict[str, Any]] = []
    for f in sidecar.get("fields") or []:
        if not isinstance(f, dict):
            continue
        name = str(f.get("id", "")).strip()
        if not name:
            continue
        ftype = f.get("type")
        info = f.get("info") or {}
        notes = info.get("notes") if isinstance(info, dict) else None
        out.append({"name": name, "type": str(ftype) if ftype else None, "notes": notes})
    return out


def sidecar_to_frame(sidecar: dict) -> pd.DataFrame:
    """Convert a CKAN sidecar's ``records`` array-of-arrays to a DataFrame.

    The ``fields[].id`` array supplies the column names. Records are
    list-of-list, so the column order in the JSON matches the column
    order in the resulting DataFrame.
    """
    fields = sidecar.get("fields") or []
    records = sidecar.get("records") or []
    col_names = [str(f.get("id", "")) for f in fields if isinstance(f, dict)]
    return pd.DataFrame(records, columns=col_names if col_names else None)


# ── Internal loader helper ───────────────────────────────────────────


def _load_one_entry(
    entry: ArsauDataset,
    *,
    data_dir: str | Path | None,
    language: str,
    allow_invalid: bool = False,
) -> RichResult:
    """Shared loader logic.  Reads the CSV from disk via the path
    cascade, attaches the sidecar schema if available, and returns
    a RichResult.

    For ``entry.is_valid is False``, raises ``ValueError`` unless
    ``allow_invalid=True``.
    """
    if not entry.is_valid and not allow_invalid:
        raise ValueError(
            f"ARSAU {entry.year_or_range!r} {entry.kind!r} is flagged "
            f"invalid by the publishing ministry ({entry.csv_filename!r}). "
            f"Pass allow_invalid=True if you really need to inspect the "
            f"rows for data-quality purposes — they MUST NOT be used in "
            f"comparative analysis."
        )

    root = resolve_data_dir("arsau", data_dir=data_dir)
    csv_path = entry.csv_path(root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ARSAU CSV not found at {csv_path!s}. The expected file "
            f"is {entry.csv_filename!r} under "
            f"{root / entry.year_or_range!s}. To fix, either:\n"
            f"  - pass data_dir=... pointing at a directory containing "
            f"{entry.year_or_range}/{entry.csv_filename}\n"
            f"  - set MORIE_ARSAU_DIR to such a directory\n"
            f"  - download the file from "
            f"https://data.ontario.ca/dataset/police-use-of-force-race-based-data"
        )

    df = pd.read_csv(csv_path, low_memory=False)

    sidecar_dict: Optional[dict] = None
    schema: Optional[DatasetSchema] = None
    sc_path = entry.sidecar_path(root)
    if sc_path is not None and sc_path.exists():
        sidecar_dict = read_sidecar(sc_path)
        schema = parse_ckan_sidecar(sc_path)

    desc = entry.description_fr if language.lower().startswith("fr") else entry.description_en

    warnings: list[str] = []
    if not entry.is_valid:
        warnings.append(
            "Ministry-flagged invalid data — do not use for comparative "
            "or quantitative analysis."
        )

    # Sanity check on shape vs registry expectations.
    if df.shape[0] != entry.expected_rows:
        warnings.append(
            f"CSV has {df.shape[0]} rows; registry expected "
            f"{entry.expected_rows}. The on-disk file may have been "
            f"refreshed upstream since this registry was built."
        )
    if df.shape[1] != entry.expected_cols:
        warnings.append(
            f"CSV has {df.shape[1]} columns; registry expected "
            f"{entry.expected_cols}."
        )

    if language.lower().startswith("fr"):
        interp = (
            f"Données ARSAU chargées: {entry.kind!r} pour "
            f"{entry.year_or_range!r}. {df.shape[0]} lignes × "
            f"{df.shape[1]} colonnes. Validité: "
            f"{'OK' if entry.is_valid else 'INVALIDE (mise en garde)'}. "
            f"{desc}"
        )
    else:
        interp = (
            f"ARSAU data loaded: {entry.kind!r} for "
            f"{entry.year_or_range!r}. {df.shape[0]} rows × "
            f"{df.shape[1]} columns. "
            f"{'Valid for analysis.' if entry.is_valid else 'INVALID — see warnings.'} "
            f"{desc}"
        )

    return RichResult(
        title=f"ARSAU {entry.year_or_range} {entry.kind}",
        call=(
            f"arsau_load_{entry.kind}(year={entry.year_or_range!r}, "
            f"language={language!r})"
        ),
        summary_lines=[
            ("Year/range", entry.year_or_range),
            ("Kind", entry.kind),
            ("Rows", df.shape[0]),
            ("Columns", df.shape[1]),
            ("Valid", "yes" if entry.is_valid else "no — invalid data"),
            ("Sidecar", "yes" if sidecar_dict is not None else "no"),
        ],
        warnings=warnings,
        interpretation=interp,
        payload={
            "data": df,
            "schema": schema,
            "sidecar": sidecar_dict,
            "year": entry.year_or_range,
            "kind": entry.kind,
            "language": language,
            "is_valid": entry.is_valid,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "csv_path": str(csv_path),
            "value": df.shape[0],
        },
    )


# ── Public loaders (one per record type) ────────────────────────────


def _coerce_year_key(year: str | int, *, range_ok: bool = False) -> str:
    """Normalise a user-supplied year/range to a key in ARSAU_REGISTRY."""
    s = str(year).strip()
    if s in ARSAU_YEARS:
        return s
    if range_ok:
        # Accept e.g. "2020-2022" or "2020to2022"
        normalised = s.replace("to", "-").replace("_", "-")
        if normalised in ARSAU_YEARS:
            return normalised
    raise ValueError(
        f"Unknown ARSAU year {year!r}. Valid keys: {ARSAU_YEARS}."
    )


def arsau_load_main_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU ``uof_main_records`` CSV for the given year.

    Available years (incident-level main records): 2023, 2024.
    The 2020-2022 release uses a different layout — see
    :func:`arsau_load_detailed_dataset` for that range.
    """
    key = _coerce_year_key(year)
    if (key, "main_records") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU main_records is not published for {key!r}; available "
            f"years for this kind: "
            f"{[y for y in ARSAU_YEARS if (y, 'main_records') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "main_records")],
        data_dir=data_dir,
        language=language,
    )


def arsau_load_individual_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU ``uof_individual_records`` CSV for the given year.

    Available years: 2023, 2024.
    """
    key = _coerce_year_key(year)
    if (key, "individual_records") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU individual_records is not published for {key!r}; "
            f"available: "
            f"{[y for y in ARSAU_YEARS if (y, 'individual_records') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "individual_records")],
        data_dir=data_dir,
        language=language,
    )


def arsau_load_probe_cycle_records(
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU ``uof_probe_cycle_records`` (CEW telemetry) CSV.

    Available years: 2023, 2024.
    """
    key = _coerce_year_key(year)
    if (key, "probe_cycle_records") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU probe_cycle_records is not published for {key!r}; "
            f"available: "
            f"{[y for y in ARSAU_YEARS if (y, 'probe_cycle_records') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "probe_cycle_records")],
        data_dir=data_dir,
        language=language,
    )


def arsau_load_weapon_records(
    year: str | int,
    *,
    allow_invalid: bool = False,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU ``uof_weapon_records`` CSV for the given year.

    Available years: 2023 (INVALID — requires ``allow_invalid=True``),
    2024.

    Parameters
    ----------
    year : str | int
        2023 or 2024.
    allow_invalid : bool, default False
        Required to be ``True`` to load the 2023 file, which the
        publishing ministry marked as invalid. See the module
        docstring for details.

    Raises
    ------
    ValueError
        If ``year == 2023`` and ``allow_invalid`` is ``False``.
    """
    key = _coerce_year_key(year)
    if (key, "weapon_records") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU weapon_records is not published for {key!r}; available: "
            f"{[y for y in ARSAU_YEARS if (y, 'weapon_records') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "weapon_records")],
        data_dir=data_dir,
        language=language,
        allow_invalid=allow_invalid,
    )


def arsau_load_aggregate_summary(
    year_range: str = "2020-2022",
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU aggregate-summary-by-year CSV.

    Currently only published as a single combined 2020-2022 file.
    """
    key = _coerce_year_key(year_range, range_ok=True)
    if (key, "aggregate_summary") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU aggregate_summary is not published for {key!r}; "
            f"available: "
            f"{[y for y in ARSAU_YEARS if (y, 'aggregate_summary') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "aggregate_summary")],
        data_dir=data_dir,
        language=language,
    )


def arsau_load_detailed_dataset(
    year_range: str = "2020-2022",
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
) -> RichResult:
    """Load the ARSAU detailed-incident-level CSV.

    Currently only published as a single combined 2020-2022 file
    (167 columns wide; the 2023+ releases switched to the
    main/individual/probe_cycle/weapon four-file split).
    """
    key = _coerce_year_key(year_range, range_ok=True)
    if (key, "detailed_dataset") not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU detailed_dataset is not published for {key!r}; "
            f"available: "
            f"{[y for y in ARSAU_YEARS if (y, 'detailed_dataset') in ARSAU_REGISTRY]}."
        )
    return _load_one_entry(
        ARSAU_REGISTRY[(key, "detailed_dataset")],
        data_dir=data_dir,
        language=language,
    )


# ── Discovery callables ─────────────────────────────────────────────


def arsau_available_years(
    *,
    data_dir: str | Path | None = None,
    language: str = "en",
) -> RichResult:
    """List ARSAU year/year-range buckets the package knows about.

    If the user has the data files on disk, the result also flags
    which entries are present vs missing in their tree.
    """
    try:
        root = resolve_data_dir("arsau", data_dir=data_dir, require_exists=False)
        root_exists = root.exists() if root else False
    except FileNotFoundError:
        root = None
        root_exists = False

    present: list[str] = []
    missing: list[str] = []
    if root_exists:
        for year in ARSAU_YEARS:
            if (root / year).exists():
                present.append(year)
            else:
                missing.append(year)

    summary_lines: list[tuple[str, Any]] = [
        ("Years/ranges (known)", ", ".join(ARSAU_YEARS)),
        ("Data root", str(root) if root else "(unset)"),
        ("Years present on disk", ", ".join(present) if present else "(none)"),
        ("Years missing", ", ".join(missing) if missing else "(none)"),
    ]

    if language.lower().startswith("fr"):
        interp = (
            f"ARSAU connaît {len(ARSAU_YEARS)} année(s) ou plage(s) d'années: "
            f"{', '.join(ARSAU_YEARS)}. "
            f"{len(present)} présente(s) sur disque, "
            f"{len(missing)} absente(s)."
        )
    else:
        interp = (
            f"ARSAU knows {len(ARSAU_YEARS)} year/range bucket(s): "
            f"{', '.join(ARSAU_YEARS)}. "
            f"{len(present)} present on disk, {len(missing)} missing."
        )

    return RichResult(
        title="ARSAU available years",
        call="arsau_available_years()",
        summary_lines=summary_lines,
        interpretation=interp,
        payload={
            "n": len(ARSAU_YEARS),
            "years": list(ARSAU_YEARS),
            "present": present,
            "missing": missing,
            "data_root": str(root) if root else None,
        },
    )


def arsau_available_datasets(
    year: str | int | None = None,
    *,
    data_dir: str | Path | None = None,
    language: str = "en",
) -> RichResult:
    """List ARSAU dataset kinds, optionally restricted to one year.

    Parameters
    ----------
    year : str | int | None
        If supplied, only list kinds available for this year/range.
        ``None`` lists everything in :data:`ARSAU_REGISTRY`.
    """
    if year is None:
        entries = list(ARSAU_REGISTRY.values())
    else:
        key = _coerce_year_key(year, range_ok=True)
        entries = [e for k, e in ARSAU_REGISTRY.items() if k[0] == key]
        if not entries:
            raise ValueError(
                f"No ARSAU datasets registered for year {year!r}."
            )

    rows: list[list[Any]] = []
    for e in entries:
        desc = e.description_fr if language.lower().startswith("fr") else e.description_en
        rows.append([
            e.year_or_range,
            e.kind,
            e.csv_filename,
            "yes" if e.is_valid else "INVALID",
            e.expected_rows,
            e.expected_cols,
            desc[:80] + ("…" if len(desc) > 80 else ""),
        ])

    if language.lower().startswith("fr"):
        interp = (
            f"{len(entries)} entrée(s) ARSAU "
            f"{'(toutes années)' if year is None else f'pour {year!r}'} "
            f"sont enregistrées dans le catalogue interne du paquet."
        )
    else:
        interp = (
            f"{len(entries)} ARSAU entry/entries "
            f"{'(all years)' if year is None else f'for {year!r}'} "
            f"are registered in the package's internal catalogue."
        )

    return RichResult(
        title=f"ARSAU available datasets{f' ({year})' if year else ''}",
        call=f"arsau_available_datasets(year={year!r})",
        summary_lines=[
            ("Entries", len(entries)),
            ("Year filter", str(year) if year else "(none)"),
        ],
        tables=[{
            "title": "Registered datasets",
            "headers": ["year", "kind", "csv", "valid", "rows", "cols", "description"],
            "rows": rows,
        }],
        interpretation=interp,
        payload={
            "n": len(entries),
            "entries": [
                {
                    "year_or_range": e.year_or_range,
                    "kind": e.kind,
                    "csv_filename": e.csv_filename,
                    "sidecar_filename": e.sidecar_filename,
                    "expected_rows": e.expected_rows,
                    "expected_cols": e.expected_cols,
                    "is_valid": e.is_valid,
                }
                for e in entries
            ],
            "value": len(entries),
        },
    )


def arsau_describe(
    kind: str,
    year: str | int,
    *,
    language: str = "en",
    data_dir: str | Path | None = None,
    n_preview_rows: int = 3,
) -> RichResult:
    """Describe a single ARSAU dataset entry.

    Returns the registry metadata, the CKAN sidecar schema if present,
    and a small preview of the actual rows if the CSV is on disk.

    Parameters
    ----------
    kind : str
        One of :data:`ARSAU_KINDS`.
    year : str | int
        One of :data:`ARSAU_YEARS`.
    language : str
        ``"en"`` or ``"fr"``.
    n_preview_rows : int, default 3
        How many rows from the CSV head to include.
    """
    key = _coerce_year_key(year, range_ok=True)
    if (key, kind) not in ARSAU_REGISTRY:
        raise ValueError(
            f"ARSAU has no {kind!r} entry for {key!r}. Use "
            f"arsau_available_datasets() to list valid combinations."
        )
    entry = ARSAU_REGISTRY[(key, kind)]

    # Resolve data dir but tolerate absence (describe should still work
    # in offline / registry-only mode).
    try:
        root = resolve_data_dir("arsau", data_dir=data_dir, require_exists=False)
    except FileNotFoundError:
        root = None

    sidecar_dict: Optional[dict] = None
    schema: Optional[DatasetSchema] = None
    preview_df: Optional[pd.DataFrame] = None
    csv_present = False

    if root is not None:
        csv_path = entry.csv_path(root)
        if csv_path.exists():
            csv_present = True
            try:
                preview_df = pd.read_csv(csv_path, nrows=n_preview_rows, low_memory=False)
            except Exception as exc:  # noqa: BLE001
                preview_df = None
        sc_path = entry.sidecar_path(root)
        if sc_path is not None and sc_path.exists():
            sidecar_dict = read_sidecar(sc_path)
            schema = parse_ckan_sidecar(sc_path)

    desc = entry.description_fr if language.lower().startswith("fr") else entry.description_en

    sections: list[dict[str, Any]] = []
    if schema is not None:
        col_rows: list[list[Any]] = []
        for col in schema.columns[:30]:
            col_rows.append([col.name, col.dtype, col.raw_type or "", (col.source_notes or "")[:60]])
        sections.append({
            "title": f"CKAN sidecar schema (first 30 of {len(schema.columns)})",
            "headers": ["column", "dtype", "ckan_type", "notes"],
            "table": col_rows,
        })
    if preview_df is not None:
        # Trim to 8 columns wide for readability in the rendered output.
        preview_cols = list(preview_df.columns[:8])
        preview_rows = [
            [str(preview_df.iloc[i][c]) if c in preview_df.columns else ""
             for c in preview_cols]
            for i in range(min(n_preview_rows, len(preview_df)))
        ]
        sections.append({
            "title": f"CSV preview (first {len(preview_rows)} rows, {len(preview_cols)} of "
                     f"{entry.expected_cols} cols)",
            "headers": preview_cols,
            "table": preview_rows,
        })

    warnings: list[str] = []
    if not entry.is_valid:
        warnings.append(
            "Ministry-flagged invalid data — do not use for "
            "comparative or quantitative analysis."
        )
    if not csv_present:
        warnings.append(
            f"CSV not present on disk under {root!s}; describe() is "
            f"showing registry + sidecar-only information."
        )

    if language.lower().startswith("fr"):
        interp = (
            f"ARSAU {entry.kind!r} pour {entry.year_or_range!r}: "
            f"{entry.expected_rows} lignes attendues × "
            f"{entry.expected_cols} colonnes. "
            f"{'Données valides.' if entry.is_valid else 'DONNÉES INVALIDES.'} "
            f"{desc}"
        )
    else:
        interp = (
            f"ARSAU {entry.kind!r} for {entry.year_or_range!r}: "
            f"{entry.expected_rows} expected rows × "
            f"{entry.expected_cols} columns. "
            f"{'Valid for analysis.' if entry.is_valid else 'INVALID — flagged.'} "
            f"{desc}"
        )

    return RichResult(
        title=f"ARSAU {entry.year_or_range} {entry.kind}",
        call=f"arsau_describe(kind={kind!r}, year={year!r})",
        summary_lines=[
            ("Year/range", entry.year_or_range),
            ("Kind", entry.kind),
            ("CSV filename", entry.csv_filename),
            ("Sidecar resource_id", entry.sidecar_filename or "(none published)"),
            ("Expected rows", entry.expected_rows),
            ("Expected cols", entry.expected_cols),
            ("Valid", "yes" if entry.is_valid else "no — invalid data"),
            ("CSV on disk", "yes" if csv_present else "no"),
        ],
        sections=sections,
        warnings=warnings,
        interpretation=interp,
        payload={
            "entry": {
                "year_or_range": entry.year_or_range,
                "kind": entry.kind,
                "csv_filename": entry.csv_filename,
                "sidecar_filename": entry.sidecar_filename,
                "expected_rows": entry.expected_rows,
                "expected_cols": entry.expected_cols,
                "is_valid": entry.is_valid,
                "description_en": entry.description_en,
                "description_fr": entry.description_fr,
            },
            "schema": schema,
            "sidecar": sidecar_dict,
            "csv_present": csv_present,
            "value": int(entry.is_valid),
        },
    )


__all__ = [
    "ARSAU_REGISTRY",
    "ARSAU_YEARS",
    "ARSAU_KINDS",
    "ArsauDataset",
    "read_sidecar",
    "sidecar_schema",
    "sidecar_to_frame",
    "arsau_load_main_records",
    "arsau_load_individual_records",
    "arsau_load_probe_cycle_records",
    "arsau_load_weapon_records",
    "arsau_load_aggregate_summary",
    "arsau_load_detailed_dataset",
    "arsau_available_years",
    "arsau_available_datasets",
    "arsau_describe",
]
