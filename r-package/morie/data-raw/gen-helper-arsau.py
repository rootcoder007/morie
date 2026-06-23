#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Generate r-package/morie/tests/testthat/helper-arsau.R from the 3
authoritative ARSAU XLSX dictionaries bundled in
r-package/morie/inst/extdata/:

    arsau_2020_2022_dictionary.json   (2 files, 175 vars)
    arsau_2023_dictionary.json        (4 files, 187 vars)
    arsau_2024_dictionary.json        (4 files, 189 vars)

Run from the package root:
    python3 data-raw/gen-helper-arsau.py

The R helper emits:
    .morie_arsau_<key>_panel(n, seed) for each of the 10 (year, kind)
        combinations in the registry, using the REAL columns + REAL
        categorical-level vocabularies from the dictionaries.
    make_synthetic_arsau(year, kind, n, seed) dispatcher.
    stage_synthetic_arsau(root) writes all 10 CSVs to the right
        <year_or_range>/<csv_filename> path layout so the ARSAU
        analyzers and loaders find them via MORIE_ARSAU_DIR.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "tests" / "testthat" / "helper-arsau.R"

# Registry: (year_or_range, kind) -> (csv_filename, dict_file_name)
# csv_filename mirrors .ARSAU_REGISTRY_LIST in R/arsau.R.
# dict_file_name is the key into the corresponding arsau_*_dictionary.json
# (case-sensitive — 2023 capitalises "UoF_", 2024 lower-cases "uof_").
REGISTRY = [
    (
        "2020-2022",
        "aggregate_summary",
        "useofforce_agrregatesummarybyyear_2020-2022.csv",
        "arsau_2020_2022",
        "UseofForce_AgrregateSummaryByYear_2020-2022.csv",
    ),
    (
        "2020-2022",
        "detailed_dataset",
        "useofforce_detaileddataset_2020-2022.csv",
        "arsau_2020_2022",
        "UseOfForce_DetailedDataset_2020-2022.csv",
    ),
    ("2023", "main_records", "uof_main_records.csv", "arsau_2023", "UoF_Main_Records"),
    ("2023", "individual_records", "uof_individual_records.csv", "arsau_2023", "UoF_Individual_Records"),
    ("2023", "probe_cycle_records", "uof_probe_cycle_records.csv", "arsau_2023", "UoF_Probe_Cycle_Records"),
    ("2023", "weapon_records", "uof_weapon_records_invaliddata.csv", "arsau_2023", "UoF_Weapon_Records"),
    ("2024", "main_records", "uof_main_records.csv", "arsau_2024", "uof_main_records"),
    ("2024", "individual_records", "uof_individual_records.csv", "arsau_2024", "uof_individual_records"),
    ("2024", "probe_cycle_records", "uof_probe_cycle_records.csv", "arsau_2024", "uof_probe_cycle_records"),
    ("2024", "weapon_records", "uof_weapon_records.csv", "arsau_2024", "uof_weapon_records"),
]


def safe_key(year: str, kind: str) -> str:
    """R identifier — `2024|main_records` -> `2024_main_records`."""
    return f"{year.replace('-', '_')}_{kind}"


def r_chr_vec(values: list[str]) -> str:
    escaped = [v.replace('"', '\\"') for v in values]
    return "c(" + ", ".join(f'"{v}"' for v in escaped) + ")"


def gen_column_expr(var: dict) -> str:
    name = var["var"]
    dtype = (var["data_type"] or "").strip()
    raw_vals = var["data_values"] or ""
    values = [v.strip() for v in raw_vals.split("\n") if v.strip()]

    # ARSAU encodes multi-select Boolean indicators as "1" / "<Blank>"
    # in the dictionary; map to integer 0/1 for the analyzers.
    if dtype == "Boolean":
        if any(v in values for v in ("1", "<Blank>")):
            return "sample(c(1L, NA_integer_), n, replace = TRUE)"
        levels = [v for v in values if v not in ("N/A", "")]
        if not levels:
            levels = ["Yes", "No"]
        return f"sample({r_chr_vec(levels)}, n, replace = TRUE)"

    if dtype == "Text":
        if values:
            return f"sample({r_chr_vec(values)}, n, replace = TRUE)"
        if name.lower().endswith("id_") or name == "BatchFileName":
            return 'sprintf("syn-%05d", sample.int(max(2L, n %/% 2L), n, replace = TRUE))'
        return "rep(NA_character_, n)"

    if dtype == "Integer":
        return "sample(0:50, n, replace = TRUE)"

    if dtype.startswith("Date"):
        return "sample(2018:2024, n, replace = TRUE)"

    if dtype == "Time":
        return (
            'sprintf("%02d:%02d:%02d", sample(0:23, n, replace = TRUE), '
            "sample(0:59, n, replace = TRUE), "
            "sample(0:59, n, replace = TRUE))"
        )

    return "rep(NA, n)"


def gen_fixture(year: str, kind: str, csv_filename: str, dict_name: str, dict_key: str, dicts: dict[str, dict]) -> str:
    key = safe_key(year, kind)
    vars_list = dicts.get(dict_name, {}).get(dict_key, [])

    # Filter trivial / non-data vars
    vars_list = [v for v in vars_list if v.get("var") and v["var"] != "_id"]

    if not vars_list:
        # Fallback: emit a single placeholder column
        return f""".morie_arsau_{key}_panel <- function(n = 60L, seed = 1L) {{
  set.seed(seed)
  data.frame(placeholder = seq_len(n), stringsAsFactors = FALSE)
}}
"""

    body_lines = []
    for v in vars_list:
        var = v["var"].strip()
        expr = gen_column_expr(v)
        body_lines.append(f"    `{var}` = {expr},")
    body = "\n".join(body_lines)
    return f""".morie_arsau_{key}_panel <- function(n = 60L, seed = 1L) {{
  set.seed(seed)
  data.frame(
{body}
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}}
"""


def main() -> int:
    extdata = ROOT / "inst" / "extdata"
    dicts = {}
    for name in ("arsau_2020_2022", "arsau_2023", "arsau_2024"):
        p = extdata / f"{name}_dictionary.json"
        if not p.is_file():
            raise SystemExit(f"missing dictionary: {p}")
        dicts[name] = json.loads(p.read_text(encoding="utf-8"))

    fixtures = []
    for year, kind, csv_filename, dict_name, dict_key in REGISTRY:
        fixtures.append(gen_fixture(year, kind, csv_filename, dict_name, dict_key, dicts))

    dispatcher_rows = ",\n".join(
        f'  list(year = "{y}", kind = "{k}", csv_filename = "{c}", helper = ".morie_arsau_{safe_key(y, k)}_panel")'
        for (y, k, c, _dn, _dk) in REGISTRY
    )

    out = f"""\
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# *** AUTO-GENERATED FILE — do not hand-edit. ***
#
# Regenerate by running:
#     python3 data-raw/gen-helper-arsau.py
#
# Source of truth: 3 bundled ARSAU XLSX dictionaries
#     inst/extdata/arsau_2020_2022_dictionary.json
#     inst/extdata/arsau_2023_dictionary.json
#     inst/extdata/arsau_2024_dictionary.json
# Each generated `.morie_arsau_<year>_<kind>_panel(n, seed)` fixture
# uses the REAL columns + REAL categorical-level vocabularies for
# that (year, kind) entry from the dictionary.
#
# `stage_synthetic_arsau(root)` writes every registered (year, kind)
# CSV under the right layout (root/<year_or_range>/<csv_filename>) so
# the existing morie_arsau_load_* loaders find them after we set
# MORIE_ARSAU_DIR = root for the test.

{"".join(fixtures)}
.morie_arsau_registry_synth <- list(
{dispatcher_rows}
)

#' Build a synthetic ARSAU data.frame matching the real (year, kind)
#' schema + categorical vocabulary.
make_synthetic_arsau <- function(year, kind, n = 60L, seed = 1L) {{
  hit <- Find(
    function(e) e$year == as.character(year) && e$kind == kind,
    .morie_arsau_registry_synth)
  if (is.null(hit)) {{
    stop(sprintf("no synthetic ARSAU fixture for year=%s kind=%s",
                 sQuote(year), sQuote(kind)),
         call. = FALSE)
  }}
  fn <- get(hit$helper, envir = environment())
  fn(n = n, seed = seed)
}}

#' Stage every registered ARSAU (year, kind) CSV under
#'   <root>/<year_or_range>/<csv_filename>
#' and return the root path. Suitable for setting MORIE_ARSAU_DIR or
#' passing as data_dir to any morie_arsau_analyze_*() call.
stage_synthetic_arsau <- function(root = tempfile("arsau_synth_"),
                                  n = 60L, seed = 1L) {{
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  for (e in .morie_arsau_registry_synth) {{
    sub <- file.path(root, e$year)
    dir.create(sub, recursive = TRUE, showWarnings = FALSE)
    df <- make_synthetic_arsau(e$year, e$kind, n = n, seed = seed)
    utils::write.csv(df, file.path(sub, e$csv_filename), row.names = FALSE)
  }}
  root
}}
"""

    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(fixtures)} (year, kind) fixtures + stager")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
