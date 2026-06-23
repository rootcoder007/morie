#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Refresh VERSION_INVENTORY.csv and DEPENDENCIES.csv.

Run from the repo root before any release (manually or via the
release-checklist).  Both CSVs are committed; this script keeps them
in lockstep with what's actually in the tree.

Outputs:
    VERSION_INVENTORY.csv  — every version-string occurrence in the repo,
                             tagged CURRENT vs HISTORICAL.
    DEPENDENCIES.csv       — every Python and R dependency with name,
                             version pin, kind (runtime/extra/suggests),
                             license, GPL-2.0-only compatibility, and notes.

The dependency-compatibility table follows the FSF license-list at
https://www.gnu.org/licenses/license-list.html.  R packages in Suggests
are marked "Yes*" because they are optional / dynamically attached and
the FSF convention is that they do not trigger GPL linking.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import tomllib

CURRENT_VERSION = None  # auto-detected below


def _detect_current_version() -> str:
    """Read pyproject.toml's [project].version field as the truth source."""
    proj = tomllib.load(open("pyproject.toml", "rb"))
    return proj["project"]["version"]


# ────────────────────────── VERSION_INVENTORY.csv ────────────────────────────

SKIP_DIRS = {".git", "node_modules", ".venv", "build", "morie.Rcheck", "__pycache__", ".Rcheck"}
SKIP_EXT = {".pyc", ".gz", ".whl", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".so", ".dylib"}


def build_version_inventory(out_path: str = "VERSION_INVENTORY.csv") -> int:
    cur = _detect_current_version()
    rows: list[dict] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(d in path.parts for d in SKIP_DIRS):
            continue
        if path.suffix in SKIP_EXT:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if not re.search(r"v?\d+\.\d+\.\d+", line):
                        continue
                    for m in re.finditer(r"v?\d+\.\d+\.\d+", line):
                        ver = m.group(0)
                        # Filter out random scientific decimals: require version-y context
                        ctx = line.lower()
                        if not any(
                            t in ctx
                            for t in [
                                "version",
                                "morie==",
                                "morie-0",
                                "v0.",
                                "v1.",
                                "release",
                                "tag",
                                "ver",
                                "morie 0",
                                "morie/",
                            ]
                        ):
                            continue
                        rows.append(
                            {
                                "file": str(path),
                                "line": i,
                                "version_match": ver,
                                "category": "CURRENT" if cur in ver else "HISTORICAL",
                                "context": line.strip()[:140],
                            }
                        )
        except Exception:
            continue

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "line",
                "version_match",
                "category",
                "context",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


# ────────────────────────── DEPENDENCIES.csv ─────────────────────────────────

# FSF-list-derived: True = combinable with GPL-2.0-only consumer.
GPL2_COMPAT = {
    "MIT": True,
    "BSD-2-Clause": True,
    "BSD-3-Clause": True,
    "BSD": True,
    "ISC": True,
    "Zlib": True,
    "Unlicense": True,
    "CC0-1.0": True,
    "GPL-2.0-only": True,
    "GPL-2.0-or-later": True,
    "LGPL-2.1-only": True,
    "LGPL-2.1-or-later": True,
    "Python-2.0": True,
    "PSF-2.0": True,
    "MPL-2.0": True,
    "HPND": True,
    "GPL-3.0-only": False,
    "GPL-3.0-or-later": False,
    "LGPL-3.0-only": False,
    "LGPL-3.0-or-later": False,
    "Apache-2.0": False,
    "ASL 2.0": False,
    "ASL2.0": False,
    "Artistic-2.0": True,
}

# Curated map: package → (license, compat-tag, notes)
CURATED: dict[str, tuple[str, str, str]] = {
    "scipy": ("BSD-3-Clause", "Yes", ""),
    "matplotlib": ("PSF-2.0", "Yes", "matplotlib custom license is BSD-style"),
    "numpy": ("BSD-3-Clause", "Yes", ""),
    "pandas": ("BSD-3-Clause", "Yes", ""),
    "scikit-learn": ("BSD-3-Clause", "Yes", ""),
    "DoubleML": ("BSD-3-Clause", "Yes", ""),
    "statsmodels": ("BSD-3-Clause", "Yes", ""),
    "rich": ("MIT", "Yes", ""),
    "httpx": ("BSD-3-Clause", "Yes", ""),
    "stamina": ("MIT", "Yes", "retry library, replaces tenacity"),
    "lifelines": ("MIT", "Yes", ""),
    "linearmodels": ("MIT", "Yes", ""),
    "patsy": ("BSD-2-Clause", "Yes", ""),
    "tabulate": ("MIT", "Yes", ""),
    "sphinx": ("BSD-2-Clause", "Yes", ""),
    "furo": ("MIT", "Yes", ""),
    "myst-parser": ("MIT", "Yes", ""),
    "pyyaml": ("MIT", "Yes", ""),
    "pyjwt": ("MIT", "Yes", ""),
    "rapidfuzz": ("MIT", "Yes", ""),
    "pytest": ("MIT", "Yes", ""),
    "pytest-cov": ("MIT", "Yes", ""),
    "textual": ("MIT", "Yes", ""),
    "dashing": ("MIT", "Yes", ""),
    "enlighten": ("MPL-2.0", "Yes", ""),
    "imbalanced-learn": ("MIT", "Yes", ""),
    "openpyxl": ("MIT", "Yes", ""),
    "beautifulsoup4": ("MIT", "Yes", ""),
    "lxml": ("BSD-3-Clause", "Yes", ""),
    "psutil": ("BSD-3-Clause", "Yes", ""),
    "pypdf": ("BSD-3-Clause", "Yes", ""),
    "codecarbon": ("MIT", "Yes", ""),
    "sphinxcontrib-mermaid": ("BSD-2-Clause", "Yes", ""),
    # R base
    "R": ("GPL-2.0-only", "Yes", "R itself"),
    "stats": ("Part of R (GPL-2.0-only)", "Yes", "ships with R itself"),
    "utils": ("Part of R (GPL-2.0-only)", "Yes", "ships with R itself"),
    "methods": ("Part of R (GPL-2.0-only)", "Yes", "ships with R itself"),
    "tools": ("Part of R (GPL-2.0-only)", "Yes", "ships with R itself"),
    "graphics": ("Part of R (GPL-2.0-only)", "Yes", "ships with R itself"),
    # R Imports
    "DBI": ("LGPL-2.1-or-later", "Yes", ""),
    "RSQLite": ("LGPL-2.1-or-later", "Yes", ""),
    "data.table": ("MPL-2.0", "Yes", ""),
    "jsonlite": ("MIT", "Yes", ""),
    "survey": ("GPL-2.0-or-later", "Yes", ""),
    "ivreg": ("GPL-2.0-or-later", "Yes", ""),
    "MatchIt": ("GPL-2.0-or-later", "Yes", ""),
    "readxl": ("MIT", "Yes", ""),
    "signal": ("GPL-2.0-or-later", "Yes", ""),
    "smotefamily": ("GPL-2.0-only", "Yes", ""),
    "DoubleML.R": ("MIT", "Yes", ""),
    # R Suggests with GPL-3 licenses (R-convention OK)
    "pracma": ("GPL-3.0-only", "Yes*", "Suggests-only, optional/not linked"),
    "mlr3": ("LGPL-3.0-only", "Yes*", "Suggests-only, optional/not linked"),
    "mlr3learners": ("LGPL-3.0-only", "Yes*", "Suggests-only, optional/not linked"),
    "dbscan": ("GPL-3.0-only", "Yes*", "Suggests-only, optional/not linked"),
    "reticulate": ("Apache-2.0", "Yes*", "Suggests-only, optional/not linked; user opt-in"),
    "testthat": ("MIT", "Yes", ""),
    "knitr": ("GPL-2.0-or-later", "Yes", ""),
    "rmarkdown": ("GPL-2.0-or-later", "Yes", ""),
}


def _parse_pep508(req: str) -> tuple[str, str]:
    m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", req.strip())
    return (m.group(1), m.group(2).strip()) if m else (req, "")


def _parse_desc_field(desc: str, field: str) -> list[str]:
    m = re.search(rf"^{field}:\s*\n?((?:\s+[^\n]+\n?)+)", desc, re.M)
    if not m:
        return []
    body = m.group(1)
    return [p.strip() for p in re.split(r",\s*\n?", body) if p.strip()]


def build_dependencies(out_path: str = "DEPENDENCIES.csv") -> int:
    rows: list[dict] = []

    pyproject = tomllib.load(open("pyproject.toml", "rb"))
    proj = pyproject.get("project", {})

    for r in proj.get("dependencies", []):
        name, version = _parse_pep508(r)
        rows.append(
            {
                "language": "Python",
                "package": name,
                "version": version or "any",
                "kind": "runtime",
                "license": "",
                "gpl2_compatible": "",
                "notes": "",
            }
        )
    for grp, deps in proj.get("optional-dependencies", {}).items():
        for r in deps:
            name, version = _parse_pep508(r)
            rows.append(
                {
                    "language": "Python",
                    "package": name,
                    "version": version or "any",
                    "kind": f"extra:{grp}",
                    "license": "",
                    "gpl2_compatible": "",
                    "notes": "",
                }
            )

    desc_path = Path("r-package/morie/DESCRIPTION")
    if desc_path.is_file():
        desc = desc_path.read_text()
        for field, kind in [
            ("Imports", "imports"),
            ("Depends", "depends"),
            ("Suggests", "suggests"),
            ("LinkingTo", "linkingto"),
        ]:
            for p in _parse_desc_field(desc, field):
                m = re.match(r"([A-Za-z0-9_.]+)\s*(?:\(([^)]+)\))?", p)
                if not m:
                    continue
                name = m.group(1)
                ver = m.group(2) or "any"
                rows.append(
                    {
                        "language": "R",
                        "package": name,
                        "version": ver,
                        "kind": kind,
                        "license": "",
                        "gpl2_compatible": "",
                        "notes": "",
                    }
                )

    for r in rows:
        cur = CURATED.get(r["package"])
        if cur:
            r["license"], r["gpl2_compatible"], r["notes"] = cur
        else:
            r["license"] = "unknown"
            r["gpl2_compatible"] = "?"
            r["notes"] = "needs manual classification"

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "language",
                "package",
                "version",
                "kind",
                "license",
                "gpl2_compatible",
                "notes",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main() -> int:
    cur = _detect_current_version()
    n_v = build_version_inventory()
    n_d = build_dependencies()
    print(f"VERSION_INVENTORY.csv: {n_v} entries (current version {cur})")
    print(f"DEPENDENCIES.csv:      {n_d} entries")

    # Quick sanity check on dependency compat
    with open("DEPENDENCIES.csv") as fh:
        compat = Counter(r["gpl2_compatible"] for r in csv.DictReader(fh))
    print("\nDependency compat:")
    for k, n in compat.most_common():
        print(f"  {k}: {n}")
    flagged = compat.get("No", 0) + compat.get("?", 0)
    if flagged:
        print(f"\n{flagged} entries need attention (see DEPENDENCIES.csv for the 'gpl2_compatible' column).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
