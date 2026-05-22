# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.dataset_dictionary — unified parser for dataset documentation.

Real-world open-data releases ship their schemas in a half-dozen
different formats, all describing the same logical thing: "what
columns does this CSV have, what type, what valid values, what does
each one mean in English and in French." This module reads those
sources and projects them onto a single in-memory schema model so
the rest of morie can validate / describe / cross-reference without
caring whether the source was Markdown, multi-sheet XLSX, or CKAN
JSON sidecar.

Sources currently supported
---------------------------

- **Markdown**: ``OTIS_DATA_DICTIONARY.md`` style — one file containing
  multiple ``### N. Dataset Title`` sections, each with a
  ``**Dataset Name**: \`b01df\``` line and a ``**Variables**`` bullet list.
- **XLSX**: official Ontario open-data dictionary releases — multi-sheet
  workbooks where one or two sheets carry English column definitions
  and one or two carry French. The exact sheet layout differs by
  publication; this module auto-detects the header row.
- **CKAN sidecar JSON**: the response payload of CKAN's
  ``datastore_search`` endpoint, saved next to a CSV. The ``fields``
  array carries column names + datastore types + optional
  ``info.notes``; ``records`` carries the data.

Public API
----------

- :class:`ColumnSpec`, :class:`DatasetSchema` — the in-memory model
- :func:`parse_markdown_dictionary`,
  :func:`parse_xlsx_dictionary`,
  :func:`parse_ckan_sidecar` — low-level parsers
- :func:`merge_schemas` — combine English + French (or MD + XLSX)
  into one bilingual schema
- :func:`validate_dataframe_against_schema` — cross-check a loaded
  DataFrame against a parsed schema; returns a :class:`RichResult`
- :func:`load_otis_dictionary`, :func:`load_arsau_dictionary` —
  high-level convenience that resolves the right files via
  :func:`morie._datapaths.resolve_data_dir` and merges
  MD + XLSX (OTIS) or English + French XLSX sheets (ARSAU)

Notes
-----

A handful of upstream artefacts — particularly in the Ontario
open-data releases — contain small annotation bugs that we surface
through :func:`validate_dataframe_against_schema` rather than fix:

- OTIS ``c05_..._race_by_region`` is the dictionary entry's filename
  but the on-disk CSV is ``c05_..._religion_by_region`` (variable
  list confirms it's religion, not race; the dict filename is a typo)
- OTIS Data Dictionary_French rows often re-state the type as
  "Booléen" where the English type was "Texte" (translation drift)
- ARSAU 2020-2022 ``File Name`` cells carry a leading space
  ("` UseOfForce_DetailedDataset`") — stripped here on read.

The point is not to fix them; the point is to make sure our parser
notices and reports them when it cross-validates against a loaded
DataFrame.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from morie._datapaths import resolve_data_dir
from morie.fn._richresult import RichResult

# ── Canonical dtype map ─────────────────────────────────────────────

#: Mapping from raw upstream type strings (English + French + CKAN) to
#: a canonical morie dtype. Lower-case keys for matching.
_DTYPE_MAP = {
    # English / canonical
    "integer": "int",
    "int": "int",
    "float": "float",
    "numeric": "float",
    "double": "float",
    "real": "float",
    "decimal": "float",
    "text": "string",
    "string": "string",
    "char": "string",
    "character": "string",
    "varchar": "string",
    "date": "date",
    "date (year)": "date",
    "datetime": "datetime",
    "timestamp": "datetime",
    "time": "string",
    "boolean": "bool",
    "bool": "bool",
    # French
    "entier": "int",
    "nombre entier": "int",
    "nombre": "float",
    "nombre réel": "float",
    "texte": "string",
    "chaîne": "string",
    "texte : chaîne": "string",  # the non-breaking-space variant is normalised
    "texte: chaîne": "string",
    "date (année)": "date",
    "booléen": "bool",
}


def _canonical_dtype(raw: str | None) -> str:
    """Map a raw type string to one of {int, float, string, date, datetime, bool}."""
    if raw is None:
        return "string"
    cleaned = str(raw).strip()
    cleaned = cleaned.replace("\xa0", " ")   # non-breaking space → space
    cleaned = re.sub(r"\s+", " ", cleaned)   # collapse repeated spaces
    return _DTYPE_MAP.get(cleaned.lower(), "string")


# ── Schema dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnSpec:
    """A single column in a dataset, projected to the canonical model."""

    name: str
    """Canonical column name as it appears in the data file."""

    dtype: str
    """One of: ``int`` | ``float`` | ``string`` | ``date`` | ``datetime`` | ``bool``."""

    description_en: str | None = None
    description_fr: str | None = None
    valid_values: tuple[str, ...] | None = None
    """Closed-set categorical values, if any. ``None`` for free-form columns."""

    nullable: bool = True
    source_notes: str | None = None
    """Free-form upstream notes (CKAN ``info.notes``, Additional Notes column,
    etc.). Retained verbatim for debugging cross-source disagreements."""

    raw_type: str | None = None
    """The exact upstream type string before canonicalisation — useful for
    surfacing the OTIS ``Texte → Booléen`` translation-drift quirk."""

    def localised_description(self, language: str = "en") -> str:
        """Return the description in the requested language, falling back gracefully."""
        if language.lower().startswith("fr") and self.description_fr:
            return self.description_fr
        if self.description_en:
            return self.description_en
        return self.description_fr or ""


@dataclass(frozen=True)
class DatasetSchema:
    """The schema of a single dataset — an ordered tuple of :class:`ColumnSpec`."""

    dataset_name: str
    """Canonical dataset identifier (e.g. ``b01df``, ``uof_main_records``)."""

    source_path: str
    """File from which this schema was parsed."""

    source_kind: str
    """``xlsx`` | ``markdown`` | ``ckan_sidecar``"""

    language: str
    """``en`` | ``fr`` | ``bilingual``"""

    columns: tuple[ColumnSpec, ...] = ()

    def find_column(self, name: str) -> ColumnSpec | None:
        """Return the ColumnSpec whose name matches ``name`` (case-insensitive)."""
        lc = name.strip().lower()
        for col in self.columns:
            if col.name.strip().lower() == lc:
                return col
        return None

    def describe(self, language: str = "en") -> str:
        """Multi-paragraph plain-language description of this schema.

        Returns
        -------
        str
            Human-readable summary: dataset name + source + column count +
            per-column one-line entries with name, type, and localised
            description.
        """
        lines = [
            f"Dataset: {self.dataset_name}",
            f"  source: {self.source_path} ({self.source_kind}, {self.language})",
            f"  columns: {len(self.columns)}",
            "",
        ]
        for col in self.columns:
            desc = col.localised_description(language) or ""
            if len(desc) > 110:
                desc = desc[:110] + "…"
            vv = ""
            if col.valid_values:
                vv = " {" + ", ".join(col.valid_values[:4]) + ("…" if len(col.valid_values) > 4 else "") + "}"
            lines.append(f"  - {col.name} [{col.dtype}]{vv}  {desc}")
        return "\n".join(lines)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


# ── Markdown parser ──────────────────────────────────────────────────


# Matches lines like  "-   `_id`: (Row ID) - Auto-generated variable..."
_MD_VAR_LINE = re.compile(
    r"^\s*-\s+`(?P<name>[^`]+)`\s*:\s*"
    r"(?:\((?P<label>[^)]*)\)\s*-?\s*)?"
    r"(?P<desc>.*?)$",
)


def parse_markdown_dictionary(path: str | Path) -> dict[str, DatasetSchema]:
    """Parse a OTIS-style Markdown data dictionary.

    The Markdown is expected to be one file with multiple
    ``### N. <Title>`` sections, each containing at least:

    - a ``**Dataset Name**: \`<id>\``` line
    - a ``**Variables**:`` heading
    - a bullet list of variable definitions

    Parameters
    ----------
    path : str | Path
        Path to the ``.md`` file.

    Returns
    -------
    dict[str, DatasetSchema]
        Keyed by ``dataset_name``.

    Examples
    --------
    >>> schemas = parse_markdown_dictionary("OTIS_DATA_DICTIONARY.md")  # doctest: +SKIP
    >>> 'b01df' in schemas                                              # doctest: +SKIP
    True
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    # Split on level-3 headings; the first piece is preamble (no dataset).
    sections = re.split(r"(?m)^###\s+\d+\.\s+.*$", text)
    titles = re.findall(r"(?m)^###\s+\d+\.\s+(.*)$", text)
    # sections[0] is preamble before any heading; pair the rest with titles.
    pairs = list(zip(titles, sections[1:]))

    schemas: dict[str, DatasetSchema] = {}
    for title, body in pairs:
        m_id = re.search(r"^\s*-\s*\*\*Dataset Name\*\*:\s*`([^`]+)`", body, re.M)
        if not m_id:
            continue
        dataset_id = m_id.group(1).strip()

        # Carve out the variables block: from "**Variables**:" until the
        # next horizontal rule "------" or the end of the section body.
        m_vars = re.search(r"\*\*Variables\*\*:\s*\n(.*?)(?:\n-{4,}\s*$|\Z)",
                            body, re.S | re.M)
        if not m_vars:
            schemas[dataset_id] = DatasetSchema(
                dataset_name=dataset_id,
                source_path=str(p),
                source_kind="markdown",
                language="en",
                columns=(),
            )
            continue

        cols: list[ColumnSpec] = []
        for line in m_vars.group(1).splitlines():
            mv = _MD_VAR_LINE.match(line)
            if not mv:
                continue
            name = mv.group("name").strip()
            label = (mv.group("label") or "").strip()
            desc = mv.group("desc").strip()
            full_en = f"{label}: {desc}".strip(": ").strip() if label else desc
            # Quick Boolean detection from the description.
            dtype = "bool" if re.search(r"\(Boolean\)", desc, re.I) else "string"
            if re.search(r"\b(Row ID|Number of|Days|Duration|Aggregate)", desc):
                dtype = "int" if "ID" not in desc else "int"
                if "Row ID" in desc:
                    dtype = "int"
            cols.append(ColumnSpec(
                name=name,
                dtype=dtype,
                description_en=full_en or None,
                source_notes=None,
                raw_type=None,
            ))

        schemas[dataset_id] = DatasetSchema(
            dataset_name=dataset_id,
            source_path=str(p),
            source_kind="markdown",
            language="en",
            columns=tuple(cols),
        )

    return schemas


# ── XLSX parser ──────────────────────────────────────────────────────


# Substrings that mark a header row (English or French).
_XLSX_HEADER_MARKERS = (
    "variable name",
    "nom de la variable",
)

# Substrings that pick out each column's role in the header. The keys
# are the canonical role; the values are substrings to search for
# (case-folded). Used to map an arbitrary upstream header layout onto
# our canonical fields.
_XLSX_COL_ROLES: dict[str, tuple[str, ...]] = {
    "file_name": ("file name", "nom du fichier"),
    "var_name": ("variable name", "nom de la variable"),
    "var_label": ("variable label", "étiquette de variable",
                  "étiquette de la variable"),
    "var_definition": ("variable definition", "définition de la variable",
                        "définition"),
    "data_values": ("data values", "type de valeur"),
    "additional_notes": ("additional notes", "notes additionnelles",
                          "notes complémentaires"),
    "data_type": ("data type", "type de données"),
    "data_format": ("data format", "format"),
    "measurement_level": ("measurement level", "niveau de mesure"),
}


def _find_header_row(rows: list[tuple[Any, ...]], max_scan: int = 12) -> int | None:
    """Locate the row index containing the column-definition headers.

    Returns the 0-indexed row number, or None if not found within the
    first ``max_scan`` rows.
    """
    for i, row in enumerate(rows[:max_scan]):
        cells = [str(c).strip().lower() for c in row if c is not None]
        joined = " | ".join(cells)
        if any(marker in joined for marker in _XLSX_HEADER_MARKERS):
            return i
    return None


def _map_header(header_row: tuple[Any, ...]) -> dict[str, int]:
    """Map the canonical role names to column indices in the header row."""
    role_to_col: dict[str, int] = {}
    for col_idx, cell in enumerate(header_row):
        if cell is None:
            continue
        text = str(cell).strip().lower().replace("\xa0", " ")
        text = re.sub(r"\s+", " ", text)
        for role, markers in _XLSX_COL_ROLES.items():
            if role in role_to_col:
                continue
            if any(m in text for m in markers):
                role_to_col[role] = col_idx
                break
    return role_to_col


def _normalise_dataset_id(raw: str) -> str:
    """Trim whitespace, drop ``.csv`` suffix, lowercase. Stable comparator."""
    s = str(raw).strip()
    # Strip a trailing ".csv" if present.
    if s.lower().endswith(".csv"):
        s = s[:-4]
    return s


def _parse_xlsx_sheet(
    sheet,
    *,
    source_path: str,
    language: str,
    header_row: int | None,
) -> dict[str, DatasetSchema]:
    """Parse one worksheet into ``{dataset_name: DatasetSchema}``.

    ``language`` should be ``"en"`` or ``"fr"`` — used to populate the
    schema language tag and decide whether to write the columns'
    descriptions into ``description_en`` or ``description_fr``.
    """
    # Materialise rows once (sheet may be in read-only mode).
    rows = [tuple(r) for r in sheet.iter_rows(values_only=True)]
    if not rows:
        return {}

    hdr_idx = header_row if header_row is not None else _find_header_row(rows)
    if hdr_idx is None:
        return {}

    header = rows[hdr_idx]
    role_to_col = _map_header(header)
    if "var_name" not in role_to_col:
        # Can't even find the variable name column; give up on this sheet.
        return {}
    if "file_name" not in role_to_col:
        # No grouping column — treat the whole sheet as one anonymous dataset.
        role_to_col["file_name"] = -1   # sentinel meaning "no grouping"

    by_dataset: dict[str, list[ColumnSpec]] = {}
    by_dataset_source_notes: dict[str, str] = {}

    for row in rows[hdr_idx + 1:]:
        # An entirely-blank row separates dataset blocks in some layouts;
        # just skip blank rows and continue.
        if all(c is None or str(c).strip() == "" for c in row):
            continue

        def get(role: str) -> str | None:
            idx = role_to_col.get(role)
            if idx is None or idx < 0 or idx >= len(row):
                return None
            v = row[idx]
            if v is None:
                return None
            s = str(v).strip()
            return s or None

        var_name = get("var_name")
        if not var_name:
            continue   # rows without a variable name aren't column definitions

        file_name = get("file_name") or "_unknown_"
        dataset_id = _normalise_dataset_id(file_name) if file_name != "_unknown_" \
                     else "_unknown_"

        # Build the description string from label + definition.
        label = get("var_label")
        defn = get("var_definition")
        notes = get("additional_notes")
        parts = []
        if label:
            parts.append(label)
        if defn:
            parts.append(defn)
        description = " — ".join(parts) if parts else None

        raw_type = get("data_type")
        dtype = _canonical_dtype(raw_type)

        # Parse valid values: usually a multi-line string.
        vv_raw = get("data_values")
        valid_values: tuple[str, ...] | None = None
        if vv_raw and vv_raw.lower() != "none":
            parts = [p.strip() for p in re.split(r"[\n,]", vv_raw) if p.strip()]
            # Skip purely numeric value ranges ("2020, 2021, 2022") only if
            # they look like an open list; otherwise keep them.
            if parts and not (len(parts) == 1 and parts[0].lower() == "none"):
                valid_values = tuple(parts)

        col_lang_kwargs: dict[str, Any] = {}
        if language == "fr":
            col_lang_kwargs["description_fr"] = description
        else:
            col_lang_kwargs["description_en"] = description

        spec = ColumnSpec(
            name=var_name,
            dtype=dtype,
            valid_values=valid_values,
            source_notes=notes,
            raw_type=raw_type,
            **col_lang_kwargs,
        )
        by_dataset.setdefault(dataset_id, []).append(spec)

    return {
        ds_id: DatasetSchema(
            dataset_name=ds_id,
            source_path=source_path,
            source_kind="xlsx",
            language=language,
            columns=tuple(cols),
        )
        for ds_id, cols in by_dataset.items()
    }


def parse_xlsx_dictionary(
    path: str | Path,
    *,
    header_row: int | None = None,
    sheet_filter: Iterable[str] | None = None,
) -> dict[str, DatasetSchema]:
    """Parse a multi-sheet XLSX data dictionary.

    Auto-detects English and French sheets by name patterns; merges
    them per dataset so each returned :class:`DatasetSchema` has both
    ``description_en`` and ``description_fr`` populated where possible.

    Parameters
    ----------
    path : str | Path
        Path to the ``.xlsx`` file.
    header_row : int | None, default None
        Force a specific 0-indexed header row. ``None`` means
        auto-detect by scanning the first ~12 rows for "Variable Name"
        / "Nom de la variable".
    sheet_filter : iterable of str, optional
        Only parse sheets whose name contains one of these substrings.
        Useful when the workbook also contains diagrams or notes
        sheets that should be skipped (e.g. the
        ``Relationships-Relations`` sheet in the 2024 ARSAU dict).

    Returns
    -------
    dict[str, DatasetSchema]
        Keyed by normalised dataset id. Each schema is bilingual when
        both an English and a French sheet describe the same dataset.

    Examples
    --------
    >>> schemas = parse_xlsx_dictionary(  # doctest: +SKIP
    ...     "/path/datadictionary_2023uof_en_fr.xlsx"
    ... )
    >>> 'uof_main_records' in {k.lower() for k in schemas}              # doctest: +SKIP
    True
    """
    import openpyxl

    p = Path(path)
    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)

    en_schemas: dict[str, DatasetSchema] = {}
    fr_schemas: dict[str, DatasetSchema] = {}

    for sheet in wb.worksheets:
        name_lc = sheet.title.lower()
        if sheet_filter and not any(s.lower() in name_lc for s in sheet_filter):
            continue

        # Skip non-data-dictionary sheets.
        if "relationship" in name_lc or "diagram" in name_lc or \
           "general notes" in name_lc or "generallnotes" in name_lc or \
           "notegénérales" in name_lc or "notesgenerales" in name_lc:
            continue

        # Decide language from sheet name.
        if any(tok in name_lc for tok in ("french", "français", "francais")):
            language = "fr"
        else:
            language = "en"

        parsed = _parse_xlsx_sheet(
            sheet,
            source_path=str(p),
            language=language,
            header_row=header_row,
        )

        target = fr_schemas if language == "fr" else en_schemas
        for ds_id, sch in parsed.items():
            # If two sheets parse the same dataset, prefer the one with
            # more populated columns (the 2023 ARSAU file has both
            # 'DictionnaireDeDonnées _Francais' and 'DD French' — same
            # dataset, different completeness).
            if ds_id not in target or len(sch.columns) > len(target[ds_id].columns):
                target[ds_id] = sch

    # Merge English + French per dataset (case-insensitive key match).
    merged: dict[str, DatasetSchema] = {}
    seen_lc: set[str] = set()
    for ds_id, sch_en in en_schemas.items():
        ds_lc = ds_id.lower()
        seen_lc.add(ds_lc)
        # Find a French schema with the same case-folded id.
        sch_fr = None
        for fr_id, fr_sch in fr_schemas.items():
            if fr_id.lower() == ds_lc:
                sch_fr = fr_sch
                break
        if sch_fr is not None:
            merged[ds_id] = merge_schemas(sch_en, sch_fr)
        else:
            merged[ds_id] = sch_en

    # Pick up French-only datasets that have no English counterpart.
    for ds_id, sch_fr in fr_schemas.items():
        if ds_id.lower() in seen_lc:
            continue
        merged[ds_id] = sch_fr

    return merged


# ── CKAN sidecar parser ──────────────────────────────────────────────


# CKAN datastore field types → canonical morie dtype.
_CKAN_TYPE_MAP = {
    "int": "int",
    "int4": "int",
    "int8": "int",
    "integer": "int",
    "bigint": "int",
    "numeric": "float",
    "float": "float",
    "float4": "float",
    "float8": "float",
    "double": "float",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "bpchar": "string",
    "timestamp": "datetime",
    "timestamptz": "datetime",
    "date": "date",
    "bool": "bool",
    "boolean": "bool",
    "json": "string",
}


def parse_ckan_sidecar(path: str | Path) -> DatasetSchema:
    """Parse a CKAN ``datastore_search`` JSON sidecar saved alongside a CSV.

    The expected payload shape is::

        {
            "fields": [
                {"id": "_id", "type": "int"},
                {"id": "col_a", "type": "text", "info": {"notes": "..."}},
                ...
            ],
            "records": [...],
        }

    Some CKAN responses wrap the above under a ``"result"`` key; this
    parser handles both shapes.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file. The filename (without ``.json``) is
        used as ``dataset_name``.

    Returns
    -------
    DatasetSchema
    """
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        payload = json.load(f)
    # Two known shapes: bare or wrapped in {"result": {...}}
    if "fields" in payload:
        fields = payload["fields"]
    elif "result" in payload and isinstance(payload["result"], dict):
        fields = payload["result"].get("fields", [])
    else:
        fields = []

    cols: list[ColumnSpec] = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = str(f.get("id", "")).strip()
        if not name:
            continue
        raw_type = f.get("type")
        dtype = _CKAN_TYPE_MAP.get(str(raw_type).lower(), "string")
        info = f.get("info") or {}
        notes = info.get("notes") if isinstance(info, dict) else None
        cols.append(ColumnSpec(
            name=name,
            dtype=dtype,
            description_en=None,    # CKAN sidecars don't carry rich descriptions
            source_notes=notes,
            raw_type=str(raw_type) if raw_type else None,
        ))

    return DatasetSchema(
        dataset_name=p.stem,
        source_path=str(p),
        source_kind="ckan_sidecar",
        language="en",
        columns=tuple(cols),
    )


# ── Schema operations ───────────────────────────────────────────────


def merge_schemas(primary: DatasetSchema, secondary: DatasetSchema) -> DatasetSchema:
    """Merge two schemas describing the same dataset.

    ``primary`` wins on conflict; ``secondary`` fills missing fields
    (description_fr, source_notes, valid_values, etc.). The merged
    schema's ``language`` is ``"bilingual"`` if the two inputs differ.

    Column matching is case-insensitive on name. Columns present in
    only one of the two schemas are kept in their original position.
    """
    by_name: dict[str, ColumnSpec] = {c.name.lower(): c for c in primary.columns}
    sec_by_name: dict[str, ColumnSpec] = {c.name.lower(): c for c in secondary.columns}

    merged_cols: list[ColumnSpec] = []
    seen: set[str] = set()
    for col in primary.columns:
        sec = sec_by_name.get(col.name.lower())
        if sec is None:
            merged_cols.append(col)
        else:
            merged_cols.append(ColumnSpec(
                name=col.name,
                dtype=col.dtype or sec.dtype,
                description_en=col.description_en or sec.description_en,
                description_fr=col.description_fr or sec.description_fr,
                valid_values=col.valid_values or sec.valid_values,
                nullable=col.nullable,
                source_notes=col.source_notes or sec.source_notes,
                raw_type=col.raw_type or sec.raw_type,
            ))
        seen.add(col.name.lower())

    for col in secondary.columns:
        if col.name.lower() not in seen:
            merged_cols.append(col)

    lang = "bilingual" if primary.language != secondary.language else primary.language

    return DatasetSchema(
        dataset_name=primary.dataset_name,
        source_path=primary.source_path,
        source_kind=primary.source_kind,
        language=lang,
        columns=tuple(merged_cols),
    )


def _coerce_dtype_match(pandas_dtype: str, canonical: str) -> bool:
    """Return True if ``pandas_dtype`` is broadly compatible with ``canonical``."""
    s = str(pandas_dtype).lower()
    if canonical == "int":
        return "int" in s
    if canonical == "float":
        return "float" in s or "int" in s   # ints are valid floats
    if canonical == "bool":
        return "bool" in s
    if canonical in ("date", "datetime"):
        return "datetime" in s or "date" in s or "object" in s
    # string / category / unknown → object / string are fine
    return "object" in s or "string" in s or "category" in s


def validate_dataframe_against_schema(
    df,
    schema: DatasetSchema,
    *,
    language: str = "en",
) -> RichResult:
    """Cross-check a loaded DataFrame against a parsed schema.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    schema : DatasetSchema
        The schema to check against.
    language : str, default "en"
        Language for the interpretation block.

    Returns
    -------
    RichResult
        ``payload`` contains:

        - ``is_valid`` (bool) — True iff missing_columns + dtype_mismatches empty
        - ``missing_columns`` (list[str]) — in schema, absent in df
        - ``extra_columns`` (list[str]) — in df, absent in schema
        - ``dtype_mismatches`` (list[dict]) — column with surprising pandas dtype
        - ``n_columns_schema`` / ``n_columns_df`` / ``n_rows_df``
    """
    schema_cols = {c.name.lower(): c for c in schema.columns}
    df_cols = {str(c).lower(): str(c) for c in df.columns}

    missing = [c.name for c in schema.columns if c.name.lower() not in df_cols]
    extra = [name for lc, name in df_cols.items() if lc not in schema_cols]

    dtype_mismatches = []
    for lc, spec in schema_cols.items():
        if lc not in df_cols:
            continue
        df_name = df_cols[lc]
        pd_dtype = str(df[df_name].dtype)
        if not _coerce_dtype_match(pd_dtype, spec.dtype):
            dtype_mismatches.append({
                "column": spec.name,
                "expected": spec.dtype,
                "actual_pandas": pd_dtype,
                "raw_upstream_type": spec.raw_type,
            })

    is_valid = not missing and not dtype_mismatches
    n_rows = len(df)
    n_cols_df = len(df.columns)
    n_cols_sch = len(schema.columns)

    # Build interpretation.
    if language.lower().startswith("fr"):
        if is_valid:
            interp = (
                f"Le DataFrame correspond au schéma {schema.dataset_name!r} : "
                f"{n_cols_df} colonnes alignées sur les {n_cols_sch} colonnes "
                f"définies; {n_rows} lignes chargées."
            )
        else:
            interp = (
                f"Le DataFrame présente des divergences par rapport au "
                f"schéma {schema.dataset_name!r} : "
                f"{len(missing)} colonne(s) attendue(s) absente(s), "
                f"{len(extra)} colonne(s) supplémentaire(s) inattendue(s), "
                f"{len(dtype_mismatches)} décalage(s) de type."
            )
    else:
        if is_valid:
            interp = (
                f"DataFrame matches schema {schema.dataset_name!r}: "
                f"{n_cols_df} columns line up with the {n_cols_sch} defined "
                f"columns; {n_rows} rows loaded."
            )
        else:
            interp = (
                f"DataFrame diverges from schema {schema.dataset_name!r}: "
                f"{len(missing)} expected column(s) missing, "
                f"{len(extra)} unexpected extra column(s), "
                f"{len(dtype_mismatches)} dtype mismatch(es). "
                f"See the per-section tables below."
            )

    sections: list[dict] = []
    if missing:
        sections.append({
            "title": "Missing columns",
            "lines": [(name, "") for name in missing],
        })
    if extra:
        sections.append({
            "title": "Extra columns (in DataFrame but not in schema)",
            "lines": [(name, "") for name in extra],
        })
    if dtype_mismatches:
        sections.append({
            "title": "Dtype mismatches",
            "headers": ["column", "expected", "actual (pandas)"],
            "table": [
                [m["column"], m["expected"], m["actual_pandas"]]
                for m in dtype_mismatches
            ],
        })

    warnings = []
    if extra and not missing:
        # Many real datasets add columns that aren't yet documented; treat
        # this as a warning, not an error.
        warnings.append(
            "DataFrame has columns not in the schema — likely the dataset "
            "has expanded since the dictionary was published."
        )

    return RichResult(
        title=f"Schema validation: {schema.dataset_name}",
        call=f"validate_dataframe_against_schema(df, schema={schema.dataset_name!r})",
        summary_lines=[
            ("Schema columns", n_cols_sch),
            ("DataFrame columns", n_cols_df),
            ("DataFrame rows", n_rows),
            ("Valid", "yes" if is_valid else "no"),
        ],
        sections=sections,
        warnings=warnings,
        interpretation=interp,
        payload={
            "is_valid": is_valid,
            "missing_columns": missing,
            "extra_columns": extra,
            "dtype_mismatches": dtype_mismatches,
            "n_columns_schema": n_cols_sch,
            "n_columns_df": n_cols_df,
            "n_rows_df": n_rows,
            "schema_name": schema.dataset_name,
        },
    )


# ── High-level convenience loaders ──────────────────────────────────


def load_otis_dictionary(
    data_dir: str | Path | None = None,
) -> dict[str, DatasetSchema]:
    """Load the OTIS data dictionary by combining its Markdown + XLSX sources.

    Resolves the OTIS data directory via
    :func:`morie._datapaths.resolve_data_dir`, so the file works
    regardless of where the user has stored their OTIS download — as
    long as it's discoverable through the documented cascade
    (``data_dir`` argument → ``MORIE_OTIS_DIR`` env var →
    ``MORIE_DATA_DIR`` + ``otis/`` → user cache → bundled fixture).

    The Markdown carries the canonical English prose; the XLSX is
    authoritative for column dtypes and valid values, plus carries
    French translations. The returned schemas are bilingual where
    both sources agree on the dataset id.

    Parameters
    ----------
    data_dir : str | Path | None, default None
        Override for the OTIS data directory. ``None`` triggers the
        env-var cascade.

    Returns
    -------
    dict[str, DatasetSchema]
        Keyed by OTIS dataset id (``b01df``, ``c12``, etc.).
    """
    root = resolve_data_dir("otis", data_dir=data_dir)

    md_path = root / "OTIS_DATA_DICTIONARY.md"
    xlsx_path = root / "od-restrictiveconfinement-segregation-deaths-dd20251103-datadictionary.xlsx"

    md_schemas: dict[str, DatasetSchema] = {}
    if md_path.exists():
        md_schemas = parse_markdown_dictionary(md_path)

    xlsx_schemas: dict[str, DatasetSchema] = {}
    if xlsx_path.exists():
        xlsx_schemas = parse_xlsx_dictionary(xlsx_path)

    # The XLSX dataset ids use filenames like
    # 'a01_restrictive_confinement_detailed_dataset' (no extension);
    # the MD uses short ids like 'b01df'. We DO NOT try to bridge them
    # by force — instead we keep both keys live, so callers can look
    # up by whichever id they have.
    merged: dict[str, DatasetSchema] = dict(xlsx_schemas)
    for md_id, md_sch in md_schemas.items():
        if md_id in merged:
            merged[md_id] = merge_schemas(merged[md_id], md_sch)
        else:
            merged[md_id] = md_sch
    return merged


# ARSAU dictionary filename per year-or-range, relative to the
# ARSAU/Suppl/ directory:
_ARSAU_DICT_FILENAMES = {
    "2020-2022": "datadictionary_useofforce_2020-2022en_fr1.xlsx",
    "2023":      "datadictionary_2023uof_en_fr.xlsx",
    "2024":      "datadictionary_uof_en_fr_v1.0_20250822.xlsx",
}


def load_arsau_dictionary(
    year_or_range: str,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, DatasetSchema]:
    """Load the ARSAU data dictionary for the given year (or year range).

    ``year_or_range`` must be one of: ``"2020-2022"``, ``"2023"``, ``"2024"``.

    The right XLSX in ``ARSAU/Suppl/`` is selected and parsed bilingually.
    """
    key = str(year_or_range).strip()
    if key not in _ARSAU_DICT_FILENAMES:
        raise ValueError(
            f"Unknown ARSAU year_or_range {year_or_range!r}; "
            f"valid keys: {sorted(_ARSAU_DICT_FILENAMES)}"
        )

    root = resolve_data_dir("arsau", data_dir=data_dir)
    xlsx_path = root / "Suppl" / _ARSAU_DICT_FILENAMES[key]
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"ARSAU dictionary for {key!r} not found at {xlsx_path}. "
            f"Expected the official Ontario data dictionary release in "
            f"{root / 'Suppl'!s}."
        )

    return parse_xlsx_dictionary(xlsx_path)


__all__ = [
    "ColumnSpec",
    "DatasetSchema",
    "parse_markdown_dictionary",
    "parse_xlsx_dictionary",
    "parse_ckan_sidecar",
    "merge_schemas",
    "validate_dataframe_against_schema",
    "load_otis_dictionary",
    "load_arsau_dictionary",
]
