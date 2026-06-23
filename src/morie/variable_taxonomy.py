# SPDX-License-Identifier: AGPL-3.0-or-later
"""morie.variable_taxonomy — classify every variable in every dataset.

Goal: drive a method dispatcher that picks the right statistical
analysis for each variable based on its formal level of measurement
(Stevens 1946: nominal / ordinal / interval / ratio), its
cardinality, its functional role (identifier / outcome / covariate /
metadata), and any cross-year-safety constraints documented in the
data dictionary.

The taxonomy is sourced from three things, in priority order:

1. **Explicit overrides** in the per-dataset registry (e.g. OTIS
   ``UniqueIndividual_ID`` is hard-coded as
   ``cross_year_safe = FALSE`` regardless of what the dictionary says,
   because the dictionary description itself states the random
   reassignment).
2. **Parsed dictionary entries** via
   :func:`morie.dataset_dictionary.load_otis_dictionary` /
   ``load_arsau_dictionary``. The dictionary supplies the raw
   ``Data Type`` (Integer, Text, Boolean, Date) and ``Data Values``
   list (closed-set valid values).
3. **Heuristics** on the column name + sample values when the
   dictionary is silent (e.g. trailing ``_Alert`` / ``_Yes_No`` ->
   nominal binary; ``Number*`` + ``*Days`` / ``*Count`` -> ratio).

Outputs are :class:`VariableTaxonomy` dataclasses, one per (dataset,
column) tuple.  They feed into the method dispatcher (built next)
and the audit walker.

References
----------
Stevens, S.S. (1946). "On the theory of scales of measurement."
  *Science*, 103(2684), 677-680.
Velleman, P.F. and Wilkinson, L. (1993). "Nominal, ordinal, interval,
  and ratio typologies are misleading." *The American Statistician*,
  47(1), 65-72.  (We acknowledge the critique; the dispatcher uses
  measurement level as a default that the caller can override per
  analysis.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from morie.dataset_dictionary import ColumnSpec, DatasetSchema

# ── Enums ──────────────────────────────────────────────────────────


class LevelOfMeasurement(str, Enum):
    """Stevens-1946 levels, plus the practical extensions we need."""

    NOMINAL = "nominal"
    """Categorical with no order (Race, Weapon, IncidentType)."""

    ORDINAL = "ordinal"
    """Categorical with order (Age_Category, severity buckets)."""

    INTERVAL = "interval"
    """Equal-interval scale, no true zero (date-as-day-number,
    temperature in C)."""

    RATIO = "ratio"
    """Equal-interval scale WITH true zero (counts, durations,
    proportions, monetary)."""

    DATE = "date"
    """Calendar date.  Treated specially because date arithmetic is
    valid but date *means* depend on calendar conventions."""

    DATETIME = "datetime"

    IDENTIFIER = "identifier"
    """ID columns (UniqueIndividual_ID, BatchFileName).  No
    statistical analysis applies directly; used for joins / grouping
    only."""

    BOOLEAN = "boolean"
    """Strict yes/no (or 1/0).  A special case of nominal binary but
    flagged because many tests are optimised for the dichotomous
    case (Wilson CI, McNemar)."""

    FREE_TEXT = "free_text"
    """Free-text narrative (rare in administrative releases).  No
    direct statistical analysis without NLP preprocessing."""

    UNKNOWN = "unknown"


class Cardinality(str, Enum):
    """Coarse cardinality classes, used to short-circuit slow methods."""

    BINARY = "binary"  # exactly 2 distinct values
    DISCRETE_LOW = "discrete_low"  # 3-10 distinct
    DISCRETE_MEDIUM = "discrete_medium"  # 11-100 distinct
    DISCRETE_HIGH = "discrete_high"  # 101-10000 distinct
    CONTINUOUS = "continuous"
    """Effectively continuous (>>1000 distinct values relative to n,
    or floating-point)."""
    UNKNOWN = "unknown"


class Role(str, Enum):
    """Functional role.  Drives whether the variable is treated as
    an identifier, an outcome to predict, a covariate to control
    for, a weight, or pure metadata."""

    IDENTIFIER = "identifier"
    OUTCOME = "outcome"
    COVARIATE = "covariate"
    WEIGHT = "weight"
    METADATA = "metadata"
    UNKNOWN = "unknown"


# ── Result type ────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariableTaxonomy:
    """The classification of a single (dataset, column) variable."""

    dataset_name: str
    column_name: str

    level: LevelOfMeasurement
    cardinality: Cardinality
    role: Role

    cross_year_safe: bool = True
    """If False, joins / aggregations across fiscal years are
    statistically invalid for this column.  OTIS
    ``UniqueIndividual_ID`` is the canonical example: random
    reassignment each fiscal year means cross-year linking is
    meaningless."""

    dictionary_described: bool = True
    """True if the column appears in the official data dictionary."""

    valid_values: tuple[str, ...] | None = None
    """Closed-set valid values from the dictionary, if any."""

    nullable: bool = True
    raw_dtype: str | None = None
    notes: str | None = None
    source: str = "unknown"  # 'dictionary' | 'heuristic' | 'override'

    def recommended_summary(self) -> str:
        """Plain-language hint at which summary statistic suits this variable.

        Used by the audit report and by ``morie_analyze_variable``.
        """
        if self.role == Role.IDENTIFIER:
            return "n distinct, top-5 frequencies (identifier — no stats)"
        if self.level == LevelOfMeasurement.IDENTIFIER:
            return "n distinct, top-5 frequencies (identifier — no stats)"
        if self.level == LevelOfMeasurement.BOOLEAN:
            return "proportion + Wilson 95% CI"
        if self.level == LevelOfMeasurement.NOMINAL:
            return "mode + frequency table + chi-square goodness-of-fit"
        if self.level == LevelOfMeasurement.ORDINAL:
            return "median + IQR + Mann-Whitney/Wilcoxon for pairwise"
        if self.level == LevelOfMeasurement.RATIO:
            return "mean + SD + Gini if non-negative + log-scale histogram + Pareto Hill-MLE if heavy-tailed"
        if self.level == LevelOfMeasurement.INTERVAL:
            return "mean + SD + skew/kurtosis + linear regression"
        if self.level in (LevelOfMeasurement.DATE, LevelOfMeasurement.DATETIME):
            return "min/max date + temporal histogram + change-point if treated as time series"
        if self.level == LevelOfMeasurement.FREE_TEXT:
            return "n distinct, length distribution (NLP otherwise)"
        return "(unknown level - manual inspection)"

    def recommended_pair_test(self, other: VariableTaxonomy) -> str:
        """Hint at the right bivariate test between this variable and ``other``.

        Levels-of-measurement → test mapping (Velleman & Wilkinson
        1993's critique noted: caller may override).
        """
        a = self.level
        b = other.level

        if a == LevelOfMeasurement.IDENTIFIER or b == LevelOfMeasurement.IDENTIFIER:
            return "(identifier — use for grouping, not for tests)"
        if a in (LevelOfMeasurement.NOMINAL, LevelOfMeasurement.BOOLEAN) and b in (
            LevelOfMeasurement.NOMINAL,
            LevelOfMeasurement.BOOLEAN,
        ):
            return "chi-square (or Fisher exact if small) + Cramer's V"
        if a == LevelOfMeasurement.ORDINAL and b == LevelOfMeasurement.ORDINAL:
            return "Spearman rho + Kendall tau"
        if {a, b} == {LevelOfMeasurement.NOMINAL, LevelOfMeasurement.INTERVAL} or {a, b} == {
            LevelOfMeasurement.NOMINAL,
            LevelOfMeasurement.RATIO,
        }:
            return "Welch's ANOVA (or Kruskal-Wallis if non-normal)"
        if {a, b} == {LevelOfMeasurement.ORDINAL, LevelOfMeasurement.INTERVAL} or {a, b} == {
            LevelOfMeasurement.ORDINAL,
            LevelOfMeasurement.RATIO,
        }:
            return "Spearman rho (rank-based) or polychoric correlation if cells are sufficient"
        if a in (LevelOfMeasurement.INTERVAL, LevelOfMeasurement.RATIO) and b in (
            LevelOfMeasurement.INTERVAL,
            LevelOfMeasurement.RATIO,
        ):
            return "Pearson r + linear regression (or Spearman if not normal)"
        if a == LevelOfMeasurement.DATE or b == LevelOfMeasurement.DATE:
            return "stratify by date bucket, then apply pair test on the rest"
        return "(unsupported pair — manual inspection)"


# ── Hard-coded invariants (override dictionary + heuristic) ────────


# (dataset_id_prefix, column_name) -> {overridden field: value}
# dataset_id_prefix matches via startswith() to capture all OTIS series.
INVARIANT_OVERRIDES: dict[tuple[str, str], dict] = {
    # OTIS: UniqueIndividual_ID is randomly reassigned every fiscal
    # year (per the data dictionary itself).  Cross-year tracking is
    # statistically invalid.
    ("b01", "UniqueIndividual_ID"): {
        "cross_year_safe": False,
        "role": Role.IDENTIFIER,
        "notes": ("Random per-fiscal-year reassignment per OTIS data dictionary; cross-year joins meaningless."),
        "source": "override",
    },
    ("a01", "UniqueIndividual_ID"): {
        "cross_year_safe": False,
        "role": Role.IDENTIFIER,
        "notes": "Random per-fiscal-year reassignment (OTIS dict).",
        "source": "override",
    },
    # ARSAU: BatchFileName + Indiv_Index together are the within-
    # incident person identifier.  Per-incident only.
    ("uof_main_records", "BatchFileName"): {
        "role": Role.IDENTIFIER,
        "notes": "Joins main↔individual↔weapon↔probe within year.",
        "source": "override",
    },
    ("uof_individual_records", "BatchFileName"): {
        "role": Role.IDENTIFIER,
        "source": "override",
    },
    ("uof_weapon_records", "BatchFileName"): {
        "role": Role.IDENTIFIER,
        "source": "override",
    },
    ("uof_probe_cycle_records", "BatchFileName"): {
        "role": Role.IDENTIFIER,
        "source": "override",
    },
    ("uof_individual_records", "Indiv_Index"): {
        "role": Role.IDENTIFIER,
        "source": "override",
    },
    # ARSAU outcome column (with trailing-space typo in 2023)
    ("uof_individual_records", "IndivInjuries_PhysicalInjuries"): {
        "role": Role.OUTCOME,
        "notes": "Yes/No -> 1/0 boolean outcome for disparity analysis.",
        "source": "override",
    },
}


def _override_for(dataset_name: str, col_name: str) -> dict | None:
    """Look up explicit overrides matching dataset prefix + column."""
    col_lc = col_name.strip().lower()
    ds_lc = dataset_name.lower()
    for (ds_prefix, col), patch in INVARIANT_OVERRIDES.items():
        if ds_lc.startswith(ds_prefix.lower()) and col_lc == col.lower():
            return patch
    return None


# ── Heuristics for cardinality + role + level when dict is silent ──


_BOOLEAN_VALUE_SETS = (
    {"yes", "no"},
    {"true", "false"},
    {"y", "n"},
    {"0", "1"},
    {"1", "0"},
)

_IDENTIFIER_PATTERNS = re.compile(
    r"(^_?id$|^id_$|_id$|[a-z]id$|^uniqueindividual|^batchfile|"
    r"^recordnum|^record_id$|^incidentnumber$|index$)",
    re.IGNORECASE,
)

_OUTCOME_NAME_PATTERNS = re.compile(
    r"(injur|death|killed|incident_outcome|outcome|fatal|"
    r"hospital|medical|treatment)",
    re.IGNORECASE,
)

_RATIO_NAME_PATTERNS = re.compile(
    r"(number|count|days|hours|minutes|seconds|cycles|"
    r"placements|reports|amount|rate|score|n_)",
    re.IGNORECASE,
)

_ORDINAL_NAME_PATTERNS = re.compile(
    r"(category|level|severity|grade|tier|rank|order)",
    re.IGNORECASE,
)


def _cardinality_from_valid_values(vv: tuple[str, ...] | None) -> Cardinality:
    if vv is None:
        return Cardinality.UNKNOWN
    n = len(vv)
    if n == 2:
        return Cardinality.BINARY
    if n <= 10:
        return Cardinality.DISCRETE_LOW
    if n <= 100:
        return Cardinality.DISCRETE_MEDIUM
    return Cardinality.DISCRETE_HIGH


def _level_from_spec(spec: ColumnSpec, dataset_name: str) -> LevelOfMeasurement:
    """Heuristic chain for inferring level of measurement."""
    name = spec.name
    dtype = spec.dtype.lower() if spec.dtype else "string"

    # 0. Explicit ID columns
    if _IDENTIFIER_PATTERNS.search(name):
        return LevelOfMeasurement.IDENTIFIER

    # 1. Dictionary-declared types
    if dtype == "bool":
        return LevelOfMeasurement.BOOLEAN
    if dtype == "date":
        return LevelOfMeasurement.DATE
    if dtype == "datetime":
        return LevelOfMeasurement.DATETIME

    # 2. Closed valid-value sets that look boolean
    if spec.valid_values is not None:
        vv_lc = {str(v).strip().lower() for v in spec.valid_values}
        for boolset in _BOOLEAN_VALUE_SETS:
            if vv_lc == boolset:
                return LevelOfMeasurement.BOOLEAN
        # Categorical with a name hint of "Category"/"Level" → ordinal
        if _ORDINAL_NAME_PATTERNS.search(name):
            return LevelOfMeasurement.ORDINAL
        # Otherwise closed-set string is nominal
        if dtype == "string":
            return LevelOfMeasurement.NOMINAL

    # 3. Name-pattern hints for count-like ratio variables
    if dtype == "int" and _RATIO_NAME_PATTERNS.search(name):
        return LevelOfMeasurement.RATIO
    if dtype == "float":
        return LevelOfMeasurement.RATIO  # default for floats

    if dtype == "int":
        # Integer without a count-like name: still ratio (counts have
        # a true zero) unless it looks ordinal
        if _ORDINAL_NAME_PATTERNS.search(name):
            return LevelOfMeasurement.ORDINAL
        return LevelOfMeasurement.RATIO

    # 4. Default: string -> nominal (could be free-text but we don't
    #    have a length distribution at classify-time)
    return LevelOfMeasurement.NOMINAL


def _role_from_spec(spec: ColumnSpec) -> Role:
    if _IDENTIFIER_PATTERNS.search(spec.name):
        return Role.IDENTIFIER
    if _OUTCOME_NAME_PATTERNS.search(spec.name):
        return Role.OUTCOME
    return Role.COVARIATE


# ── Public classifier ──────────────────────────────────────────────


def classify_variable(
    spec: ColumnSpec,
    *,
    dataset_name: str,
) -> VariableTaxonomy:
    """Classify a single variable given its dictionary entry.

    Parameters
    ----------
    spec : ColumnSpec
        The parsed column entry from
        :mod:`morie.dataset_dictionary`.
    dataset_name : str
        The owning dataset id (e.g. ``"b01"`` for OTIS,
        ``"uof_main_records"`` for ARSAU).

    Returns
    -------
    VariableTaxonomy
    """
    level = _level_from_spec(spec, dataset_name)
    role = _role_from_spec(spec)
    card = _cardinality_from_valid_values(spec.valid_values)
    if card == Cardinality.UNKNOWN and level == LevelOfMeasurement.BOOLEAN:
        card = Cardinality.BINARY
    if level == LevelOfMeasurement.RATIO and card == Cardinality.UNKNOWN:
        card = Cardinality.CONTINUOUS
    if level == LevelOfMeasurement.IDENTIFIER:
        card = Cardinality.DISCRETE_HIGH

    tax = VariableTaxonomy(
        dataset_name=dataset_name,
        column_name=spec.name,
        level=level,
        cardinality=card,
        role=role,
        cross_year_safe=True,
        dictionary_described=True,
        valid_values=spec.valid_values,
        nullable=spec.nullable,
        raw_dtype=spec.raw_type,
        notes=None,
        source="dictionary" if spec.description_en or spec.description_fr else "heuristic",
    )

    # Apply hard-coded overrides last.
    override = _override_for(dataset_name, spec.name)
    if override:
        # Build a kwargs dict from the existing taxonomy + patch.
        from dataclasses import replace

        kwargs = {k: v for k, v in override.items() if k != "notes"}
        tax = replace(tax, **kwargs, notes=override.get("notes"))

    return tax


def classify_schema(schema: DatasetSchema) -> list[VariableTaxonomy]:
    """Classify every column in a :class:`DatasetSchema`."""
    return [classify_variable(col, dataset_name=schema.dataset_name) for col in schema.columns]


__all__ = [
    "LevelOfMeasurement",
    "Cardinality",
    "Role",
    "VariableTaxonomy",
    "classify_variable",
    "classify_schema",
    "INVARIANT_OVERRIDES",
]
