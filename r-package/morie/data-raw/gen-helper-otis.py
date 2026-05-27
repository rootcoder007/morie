#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate r-package/morie/tests/testthat/helper-otis.R from the
authoritative OTIS data dictionary at
r-package/morie/inst/extdata/otis_dictionary.json.

Run this from the package root:
    python3 data-raw/gen-helper-otis.py

This is the single source of truth for the OTIS synthetic-data fixtures
used in tests/testthat/test-otis_all_analyze.R. The dictionary itself
is extracted from the Ontario MCSCS publication
`od-restrictiveconfinement-segregation-deaths-dd20251103-datadictionary.xlsx`
(see audit/otis-dictionary-extract.json + the audit transcript).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # r-package/morie
DICT_PATH = ROOT / "inst" / "extdata" / "otis_dictionary.json"
OUT_PATH = ROOT / "tests" / "testthat" / "helper-otis.R"


def r_chr_vec(values: list[str]) -> str:
    """Render an R character vector literal."""
    escaped = [v.replace('"', '\\"') for v in values]
    quoted = ", ".join(f'"{v}"' for v in escaped)
    return f"c({quoted})"


def gen_column_expr(var: dict) -> str:
    """Build the R RHS expression for one variable column in a fixture."""
    name = var["var"]
    dtype = var["data_type"].strip()
    raw_vals = var["data_values"] or ""
    # Dictionary lists multi-line value lists; split + strip empties
    values = [v.strip() for v in raw_vals.split("\n") if v.strip()]

    # Booleans encoded as Yes/No text per dictionary
    if dtype == "Boolean":
        levels = values if values else ["Yes", "No"]
        # Drop sentinel/missing levels (N/A, empty) so analyzers see Yes/No
        levels = [v for v in levels if v not in ("N/A", "")]
        if not levels:
            levels = ["Yes", "No"]
        return f"sample({r_chr_vec(levels)}, n, replace = TRUE)"

    if dtype == "Text":
        if values:
            return f"sample({r_chr_vec(values)}, n, replace = TRUE)"
        # Free-text id field — generate predictable IDs
        if name.endswith("_ID"):
            return ('sprintf("syn-%05d", '
                    "sample.int(max(2L, n %/% 4L), n, replace = TRUE))")
        return f"rep(NA_character_, n)"

    if dtype == "Integer":
        # Counts/durations — modest non-negative integer range
        return "sample(0:80, n, replace = TRUE)"

    if dtype.startswith("Date"):
        # Single fiscal-year integer (YYYY)
        return "sample(2018:2024, n, replace = TRUE)"

    # Anything else: NA placeholder so the column at least exists
    return "rep(NA, n)"


def gen_fixture(short_id: str, vars: list[dict]) -> str:
    """Emit one .morie_otis_<id>_panel(n, seed) R function."""
    body_lines: list[str] = []
    for v in vars:
        # Skip the auto-generated _id column (not present in CSVs)
        if v["var"] == "_id":
            continue
        expr = gen_column_expr(v)
        body_lines.append(f"    {v['var']} = {expr},")

    # Trim trailing comma on the last data.frame argument; add stringsAsFactors
    body = "\n".join(body_lines)
    return f""".morie_otis_{short_id}_panel <- function(n = 200L, seed = 1L) {{
  set.seed(seed)
  data.frame(
{body}
    stringsAsFactors = FALSE
  )
}}
"""


def main() -> int:
    if not DICT_PATH.is_file():
        print(f"ERROR: dictionary not found at {DICT_PATH}", file=sys.stderr)
        return 2
    dictionary: dict[str, list[dict]] = json.loads(
        DICT_PATH.read_text(encoding="utf-8"))

    # Sort ids so the generated file is deterministic
    ids = sorted(dictionary.keys())

    fixtures = [gen_fixture(sid, dictionary[sid]) for sid in ids]

    header = """\
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# *** AUTO-GENERATED FILE — do not hand-edit. ***
#
# Regenerate by running:
#     python3 data-raw/gen-helper-otis.py
#
# Source of truth: inst/extdata/otis_dictionary.json (extracted from
# the Ontario MCSCS XLSX dictionary
# od-restrictiveconfinement-segregation-deaths-dd20251103-datadictionary.xlsx).
#
# Per-dataset OTIS synthetic-fixture dispatcher. Each OTIS publication
# id has its own column schema + categorical-level vocabulary; feeding
# a one-size-fits-all panel to every analyzer triggered silent
# tryCatch+skip bail-outs and left ~50% of otis_all_analyze.R unreached.
# This helper draws every column name + every categorical level from
# the authoritative dictionary.

"""

    dispatcher = """\

# a01 mirrors b01 (person-level placements) per Python parity. The
# dictionary lists a01 separately, but if the analyzer expects b01-style
# columns, we fall through to b01 via the dispatcher when ids overlap.

#' Build a synthetic OTIS data.frame matching the schema of the
#' published dataset with the given id.
make_synthetic_otis <- function(id, n = 200L, seed = 1L) {
  helper_name <- paste0(".morie_otis_", id, "_panel")
  if (!exists(helper_name, envir = environment())) {
    # Last-resort fallback: b01 person-level panel.
    return(.morie_otis_b01_panel(n = n, seed = seed))
  }
  fn <- get(helper_name, envir = environment())
  fn(n = n, seed = seed)
}

#' Return the full datasets-list (a01 + b01..b09 + c01..c12 + d01..d07)
#' with each entry built from its proper schema.
make_synthetic_otis_datasets_complete <- function(n = 80L, seed = 2L) {
  ids <- c("a01",
           paste0("b0", 1:9),
           paste0("c0", 1:9), "c10", "c11", "c12",
           paste0("d0", 1:7))
  setNames(
    lapply(seq_along(ids), function(i)
      make_synthetic_otis(ids[i], n = n, seed = seed + i)),
    ids
  )
}
"""

    OUT_PATH.write_text(header + "\n".join(fixtures) + dispatcher,
                        encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(ids)} datasets, "
          f"{sum(len(v) for v in dictionary.values())} variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
