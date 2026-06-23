#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate r-package/morie/tests/testthat/helper-tps.R from the
authoritative TPS dictionary at
r-package/morie/inst/extdata/tps_dictionary.json.

Run from the package root:
    python3 data-raw/gen-helper-tps.py

The dictionary itself is extracted from
data/datasets/TPS/PSDP_Open_Data_Documentation.pdf plus the actual
per-category CSV headers (sampled categorical levels for low-cardinality
columns) by audit/dictionaries/tps_dictionary.json.

Beyond the synthetic-data dispatcher this generator also emits two
loader stubs (morie_tps_load_tps_dataset / morie_tps_load_tps) that the
data-seeded analyzers in tps_statphysics.R / tps_hawkes_advanced.R
check for via exists(). With the stubs in place they actually run on
synthetic Toronto-shaped data instead of short-circuiting at the
NotYetPorted check, lifting their covr.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "inst" / "extdata" / "tps_dictionary.json"
OUT_PATH = ROOT / "tests" / "testthat" / "helper-tps.R"

# Toronto bounding box used by the TPS analyzers (LAT_WGS84/LONG_WGS84
# range coded into tps_statphysics.R::morie_tps_levy_flight_alpha).
LAT_MIN, LAT_MAX = 43.58, 43.88
LON_MIN, LON_MAX = -79.62, -79.13
YEAR_MIN, YEAR_MAX = 2014, 2026


def r_chr_vec(values: list[str]) -> str:
    escaped = [v.replace('"', '\\"') for v in values]
    return "c(" + ", ".join(f'"{v}"' for v in escaped) + ")"


def gen_column_expr(col: str, cats: dict[str, list[str]]) -> str:
    """Return the R RHS for one column of a TPS synthetic panel."""
    if col in cats and len(cats[col]) >= 2:
        # Sampled distinct values from the real CSV
        return f"sample({r_chr_vec(cats[col])}, n, replace = TRUE)"

    # Domain-specific generators for well-known column names
    if col in ("OBJECTID", "OBJECTID_1", "Id_"):
        return "seq_len(n)"
    if col in ("EVENT_UNIQUE_ID", "INDEX"):
        return 'sprintf("evt-%05d", sample.int(max(2L, n %/% 2L), n, replace = TRUE))'
    if col == "LAT_WGS84":
        return f"stats::runif(n, min = {LAT_MIN}, max = {LAT_MAX})"
    if col == "LONG_WGS84":
        return f"stats::runif(n, min = {LON_MIN}, max = {LON_MAX})"
    if col in ("x", "y"):
        return "stats::runif(n, min = -1, max = 1)"
    if col.endswith("_YEAR") or col == "REPORTED_YEAR" or col == "OCCURRENCE_YEAR":
        return f"sample({YEAR_MIN}:{YEAR_MAX}, n, replace = TRUE)"
    if col.endswith("_MONTH"):
        return (
            'sample(c("January","February","March","April","May","June",'
            '"July","August","September","October","November","December"),'
            " n, replace = TRUE)"
        )
    if col.endswith("_DOW"):
        return 'sample(c("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"), n, replace = TRUE)'
    if col.endswith("_DOY"):
        return "sample(1:366, n, replace = TRUE)"
    if col.endswith("_DAY"):
        return "sample(1:31, n, replace = TRUE)"
    if col.endswith("_HOUR"):
        return "sample(0:23, n, replace = TRUE)"
    if col.endswith("_DATE") or col in ("OCCURRENCE_DATE", "REPORTED_DATE"):
        return (
            'format(as.POSIXct("2014-01-01 00:00:00", tz = "UTC") + '
            "sample.int(86400L * 365L * 12L, n, replace = TRUE), "
            '"%Y-%m-%dT%H:%M:%S.000Z")'
        )
    if col in ("OCCURRENCE_TIME", "REPORTED_TIME"):
        return (
            'sprintf("%02d:%02d:%02d", sample(0:23, n, replace = TRUE), '
            "sample(0:59, n, replace = TRUE), "
            "sample(0:59, n, replace = TRUE))"
        )
    if col.startswith("HOOD_"):
        return "sample(sprintf('%03d', 1:158), n, replace = TRUE)"
    if col.startswith("NEIGHBOURHOOD"):
        return (
            "sample(c('West Humber-Clairville', 'Mount Olive-Silverstone-"
            "Jamestown', 'Thistletown-Beaumond Heights', 'Rexdale-Kipling',"
            " 'Elms-Old Rexdale', 'Kingsview Village-The Westway',"
            " 'Willowridge-Martingrove-Richview'), n, replace = TRUE)"
        )
    if col == "DIVISION":
        return (
            "sample(c('D11','D12','D13','D14','D22','D23','D31','D32',"
            "'D33','D41','D42','D43','D51','D52','D53','D54','D55'),"
            " n, replace = TRUE)"
        )
    if col in ("REPORT_DATE",):
        return (
            'format(as.POSIXct("2014-01-01", tz = "UTC") + '
            "sample.int(86400L * 365L * 12L, n, replace = TRUE), "
            '"%Y-%m-%dT%H:%M:%S.000Z")'
        )
    if col == "OCC_DATE":
        return (
            'format(as.POSIXct("2014-01-01", tz = "UTC") + '
            "sample.int(86400L * 365L * 12L, n, replace = TRUE), "
            '"%Y-%m-%dT%H:%M:%S.000Z")'
        )

    # Generic counts / rates
    if (
        col.startswith("ASSAULT_")
        or col.startswith("ROBBERY_")
        or col.startswith("BREAKENTER_")
        or col.startswith("HOMICIDE_")
        or col.startswith("AUTOTHEFT_")
        or col.endswith("_RATE")
        or col.startswith("THEFTFROMMV_")
    ):
        return "stats::runif(n, min = 0, max = 200)"
    if col in ("COUNT", "FATALITIES", "INJURIES"):
        return "sample(0:5, n, replace = TRUE)"

    # Default: NA placeholder so the column at least exists
    return "rep(NA_character_, n)"


def gen_fixture(cat: str, headers: list[str], cats: dict[str, list[str]]) -> str:
    body_lines = []
    for col in headers:
        # Skip null / empty / BOM-only headers
        if not col or col.strip() == "":
            continue
        # Strip a UTF-8 BOM if present
        clean = col.lstrip("﻿")
        expr = gen_column_expr(clean, cats)
        body_lines.append(f"    `{clean}` = {expr},")
    body = "\n".join(body_lines)
    return f""".morie_tps_{cat}_panel <- function(n = 200L, seed = 1L) {{
  set.seed(seed)
  data.frame(
{body}
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}}
"""


def main() -> int:
    if not DICT_PATH.is_file():
        raise SystemExit(f"missing dictionary: {DICT_PATH}")
    d: dict[str, dict] = json.loads(DICT_PATH.read_text(encoding="utf-8"))

    fixtures = []
    for cat in sorted(d.keys()):
        headers = d[cat]["headers"]
        cats = d[cat]["categoricals"]
        fixtures.append(gen_fixture(cat, headers, cats))

    # Lower-case alias map (analyzers pass "Assault", canonicalization in
    # tps_io.R is case-insensitive; the analyzers ultimately accept
    # category strings as-typed)
    alias_lines = ",\n".join(f'  "{cat.lower()}" = "{cat}"' for cat in sorted(d.keys()))

    out = f"""\
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# *** AUTO-GENERATED FILE — do not hand-edit. ***
#
# Regenerate by running:
#     python3 data-raw/gen-helper-tps.py
#
# Source of truth: inst/extdata/tps_dictionary.json (extracted from
# data/datasets/TPS/PSDP_Open_Data_Documentation.pdf plus the actual
# per-category CSV headers + sampled categorical levels — see
# data-raw/gen-helper-tps.py header for the full provenance chain).
#
# Per-category TPS synthetic-data dispatcher. Each TPS dataset has its
# own column schema; the data-seeded analyzers in tps_statphysics.R and
# tps_hawkes_advanced.R check for exists("morie_tps_load_tps_dataset")
# and short-circuit if absent. By defining that loader stub here against
# globalenv() the analyzers find it via the exists() lookup and run on
# dictionary-driven synthetic Toronto data instead of NotYetPorted-ing.

{"".join(fixtures)}
.morie_tps_canonical_map <- list(
{alias_lines}
)

.morie_tps_canonical_internal <- function(name) {{
  k <- tolower(as.character(name))
  if (k %in% names(.morie_tps_canonical_map))
    return(.morie_tps_canonical_map[[k]])
  # Fall back to first registered category so the loader never NULL-fails
  .morie_tps_canonical_map[[1L]]
}}

#' Build a synthetic TPS data.frame matching the schema + categorical
#' vocabulary of the given category (case-insensitive).
make_synthetic_tps <- function(category, n = 1000L, seed = 1L) {{
  cat <- .morie_tps_canonical_internal(category)
  helper_name <- paste0(".morie_tps_", cat, "_panel")
  fn <- get(helper_name, envir = environment())
  fn(n = n, seed = seed)
}}

# ---- loader stubs ----------------------------------------------------
#
# tps_statphysics.R and tps_hawkes_advanced.R call exists("morie_tps_-
# load_tps_dataset", mode = "function") to decide whether to dispatch
# the data-seeded path or stop with "NotYetPorted". Assigning the
# loader into globalenv() satisfies the exists() check and lets the
# analyzers exercise their core algorithms during testing.

if (!exists("morie_tps_load_tps_dataset", envir = globalenv(),
            mode = "function", inherits = FALSE)) {{
  assign("morie_tps_load_tps_dataset",
         function(category, nrows = NULL) {{
           n <- if (is.null(nrows)) 1000L else as.integer(nrows)
           make_synthetic_tps(category, n = n, seed = 1L)
         }},
         envir = globalenv())
}}

if (!exists("morie_tps_load_tps", envir = globalenv(),
            mode = "function", inherits = FALSE)) {{
  assign("morie_tps_load_tps",
         function(category, nrows = NULL, format = "csv", ...) {{
           # `format` accepted but ignored — synthetic frames carry the
           # canonical columns regardless of physical export format.
           n <- if (is.null(nrows)) 1000L else as.integer(nrows)
           make_synthetic_tps(category, n = n, seed = 1L)
         }},
         envir = globalenv())
}}
"""

    OUT_PATH.write_text(out, encoding="utf-8")
    n_fix = len(fixtures)
    print(f"Wrote {OUT_PATH}")
    print(f"  {n_fix} per-category fixtures + 2 loader stubs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
