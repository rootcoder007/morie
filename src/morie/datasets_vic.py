# SPDX-License-Identifier: AGPL-3.0-or-later
"""Victorian crime data (Crime Statistics Agency, Victoria, Australia).

The agency publishes one multi-sheet ``.xlsx`` workbook per topic each
quarter. The files are not bundled -- roughly 65 MB per release, and
refreshed quarterly -- so the catalog below carries the download URLs
and the loader caches to disk, mirroring ``R/datasets_vic.R``.

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
Attribution: Crime Statistics Agency, Victoria.
Source: https://www.crimestatistics.vic.gov.au/crime-statistics/
        latest-victorian-crime-data/download-data
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from morie.fn import _frame_core as pd

__all__ = [
    "vic_catalog",
    "vic_cache_dir",
    "vic_table",
    "vic_sheets",
    "vic_offence_trend",
    "vic_lga_rates",
    "vic_indigenous_ratio",
]

_CATALOG_JSON = Path(__file__).with_name("datasets_vic_catalog.json")


def vic_catalog():
    """Return the catalog of published workbooks as a list of dicts."""
    with open(_CATALOG_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def vic_cache_dir(cache_dir=None):
    """Cache directory for downloaded workbooks, created if absent."""
    if cache_dir is None:
        cache_dir = os.environ.get("MORIE_VIC_CACHE") or ""
    if not cache_dir:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "morie", "vic")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _entry(key):
    for e in vic_catalog():
        if e["key"] == key:
            return e
    raise ValueError("unknown key %r; see vic_catalog()" % (key,))


def vic_table(key, table=1, cache_dir=None, offline=True):
    """Read one sheet of a Victorian workbook.

    ``table`` is an integer table number (1 gives "Table 01") or a sheet
    name. With ``offline=True`` (the default) nothing is downloaded: a
    cached file is read, otherwise an empty frame comes back, so tests
    never touch the network.
    """
    e = _entry(key)
    dest = os.path.join(vic_cache_dir(cache_dir), e["file"])
    if not os.path.exists(dest):
        if offline:
            return pd.DataFrame({})
        import urllib.request
        req = urllib.request.Request(
            e["url"], headers={"User-Agent": "morie/dataset-loader"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
            fh.write(r.read())
    sheet = ("Table %02d" % int(table)) if isinstance(table, int) else table
    return pd.read_excel(dest, sheet_name=sheet)


def vic_sheets(key, cache_dir=None):
    """Sheet names in a cached workbook, or [] when it is not cached."""
    e = _entry(key)
    dest = os.path.join(vic_cache_dir(cache_dir), e["file"])
    if not os.path.exists(dest):
        return []
    return list(pd.ExcelFile(dest).sheet_names)


def _need(df, cols):
    missing = [c for c in cols if c not in list(df.columns)]
    if missing:
        raise ValueError("missing column(s): %s" % ", ".join(missing))


def vic_offence_trend(data):
    """Incidents by year and offence division, with the change over the
    observed window. Mirrors ``morie_vic_offence_trend``."""
    _need(data, ["Year", "Offence Division", "Incidents Recorded"])
    rows = list(zip([float(v) for v in data["Year"]],
                    [str(v) for v in data["Offence Division"]],
                    [float(v) for v in data["Incidents Recorded"]]))
    agg = {}
    for y, d, n in rows:
        agg[(d, y)] = agg.get((d, y), 0.0) + n
    out = []
    for d in sorted({k[0] for k in agg}):
        ys = sorted(y for (dd, y) in agg if dd == d)
        if not ys:
            continue
        f, l = agg[(d, ys[0])], agg[(d, ys[-1])]
        out.append({
            "division": d, "first_year": ys[0], "last_year": ys[-1],
            "first_count": f, "last_count": l, "abs_change": l - f,
            "pct_change": (100.0 * (l - f) / f) if f > 0 else float("nan"),
        })
    return pd.DataFrame({k: [r[k] for r in out] for k in (
        "division", "first_year", "last_year", "first_count",
        "last_count", "abs_change", "pct_change")}) if out else pd.DataFrame({})


def vic_lga_rates(data, year=None, top=10):
    """Local government areas ranked by offence rate."""
    _need(data, ["Year", "Local Government Area", "Incidents Recorded",
                 "Rate per 100,000 population"])
    rows = list(zip([float(v) for v in data["Year"]],
                    [str(v) for v in data["Local Government Area"]],
                    [float(v) for v in data["Incidents Recorded"]],
                    [float(v) for v in data["Rate per 100,000 population"]]))
    if not rows:
        return pd.DataFrame({})
    target = max(r[0] for r in rows) if year is None else float(year)
    sub = [r for r in rows if r[0] == target]
    sub.sort(key=lambda r: -r[3])
    sub = sub[:int(top)]
    return pd.DataFrame({
        "lga": [r[1] for r in sub],
        "year": [r[0] for r in sub],
        "incidents": [r[2] for r in sub],
        "rate": [r[3] for r in sub],
        "rank": list(range(1, len(sub) + 1)),
    })


def vic_indigenous_ratio(data, by="Offence Division", year=None):
    """Indigenous vs non-Indigenous victim COUNT ratio, per group.

    Not a population-adjusted rate ratio: the published table carries
    counts only. The table's "Total People" rows are excluded -- folding
    them into the non-Indigenous count double-counts every victim -- and
    one year is used at a time.
    """
    _need(data, [by, "Indigenous Status", "Victim Reports"])
    years = ([float(v) for v in data["Year"]]
             if "Year" in list(data.columns) else None)
    grp = [str(v) for v in data[by]]
    status = [str(v) for v in data["Indigenous Status"]]
    n = [float(v) for v in data["Victim Reports"]]
    if years:
        target = max(years) if year is None else float(year)
        keep = [i for i, y in enumerate(years) if y == target]
        grp = [grp[i] for i in keep]
        status = [status[i] for i in keep]
        n = [n[i] for i in keep]
    agg = {}
    for g, s, v in zip(grp, status, n):
        if "Total" in s:
            continue
        a, b = agg.get(g, (0.0, 0.0))
        if "Aboriginal" in s:
            a += v
        elif "Non-Indigenous" in s:
            b += v
        agg[g] = (a, b)
    keys = sorted(agg)
    if not keys:
        return pd.DataFrame({})
    return pd.DataFrame({
        "group": keys,
        "indigenous": [agg[k][0] for k in keys],
        "non_indigenous": [agg[k][1] for k in keys],
        "count_ratio": [(agg[k][0] / agg[k][1]) if agg[k][1] > 0
                        else float("nan") for k in keys],
    })
