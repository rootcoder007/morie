# SPDX-License-Identifier: AGPL-3.0-or-later
"""Curated datasets at data.rmorie.com, opened by the MORIE key.

The MORIE project materialises public BigQuery datasets (Chicago crime, EPA
air quality, census, FEC, FDA, NOAA, ...) and serves them as CSV from the
edge (Cloudflare R2 behind a Worker), so they are reachable whenever the
internet is, with the same key ``morie login`` stores for the hosted model
tier. ``/manifest.json`` lists every table with its rows, columns, size,
SHA-256 and the BigQuery source it was built from; ``/<db>/<table>.csv.gz``
is the table. Downloads are cached in the dataset store, so a key is
needed once per table, not per call.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from morie.fn import _frame_core as pd

DEFAULT_DATA_URL = "https://data.rmorie.com"
_MANIFEST_TTL = 24 * 3600


class DataHubAuthError(RuntimeError):
    """No key, or a key the gateway rejects."""


def data_url() -> str:
    return os.environ.get("MORIE_DATA_URL", DEFAULT_DATA_URL).rstrip("/")


def _key() -> str | None:
    from .hosted import hosted_key

    return hosted_key()


def _manifest_cache_path() -> Path:
    from .data import _user_cache_dir

    return _user_cache_dir() / "data_rmorie_manifest.json"


def _open(path: str, timeout: int = 60):
    key = _key()
    if not key:
        raise DataHubAuthError(
            "data.rmorie.com needs your MORIE key: run `morie login` (GitHub) or `morie login --email you@example.com` once (R: `rmorie login`)."
        )
    req = Request(
        data_url() + path, headers={"Authorization": f"Bearer {key}", "User-Agent": "morie/1 (+https://rmorie.com)"}
    )
    try:
        return urlopen(req, timeout=timeout)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise DataHubAuthError("data.rmorie.com rejected the stored key; run `morie login` again.") from exc
        raise


def _get(path: str, timeout: int = 60) -> bytes:
    with _open(path, timeout) as resp:
        return resp.read()


def _get_to_file(path: str, dest: Path, label: str, timeout: int = 600) -> int:
    """Stream a gateway file to ``dest`` with progress; a dropped transfer is resumed."""
    from ._progress import download_url

    key = _key()
    if not key:
        raise DataHubAuthError(
            "data.rmorie.com needs your MORIE key: run `morie login` (GitHub) or `morie login --email you@example.com` once (R: `rmorie login`)."
        )
    try:
        return download_url(
            data_url() + path,
            dest,
            label,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}", "User-Agent": "morie/1 (+https://rmorie.com)"},
            opener=lambda req, timeout: urlopen(req, timeout=timeout),
        )
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise DataHubAuthError("data.rmorie.com rejected the stored key; run `morie login` again.") from exc
        raise


def hosted_manifest(refresh: bool = False) -> dict:
    """The gateway's manifest, cached for a day under the user cache dir."""
    p = _manifest_cache_path()
    if not refresh and p.exists() and time.time() - p.stat().st_mtime < _MANIFEST_TTL:
        return json.loads(p.read_text(encoding="utf-8"))
    raw = _get("/manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def cached_manifest() -> dict | None:
    """The manifest if it was fetched before (no network, no key needed)."""
    p = _manifest_cache_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def hosted_table_name(key: str) -> str:
    return "hub_" + key.replace("/", "__")


def is_hosted_key(key: str) -> bool:
    return "/" in key and not key.startswith(("/", ".")) and not key.endswith("/")


def load_hosted_dataset(key: str, *, db_path: str | Path | None = None, refresh: bool = False):
    """``db/table`` from data.rmorie.com: the dataset store first, then one download."""
    from .data import cache_load, cache_store

    if not is_hosted_key(key):
        raise KeyError(f"{key!r} is not a data.rmorie.com key (expected db/table; see `morie list-datasets`)")
    table = hosted_table_name(key)
    if not refresh:
        try:
            cached = cache_load(table, db_path)
        except Exception:  # noqa: BLE001 - a broken cache must not block the download
            cached = None
        if cached is not None and len(cached) > 0:
            return cached  # an empty cached table is a miss: the edge copy may have been rebuilt
    _check_known(key)
    db, tbl = key.split("/", 1)
    # streamed to disk, not held in memory: a table can be a gigabyte compressed
    with tempfile.TemporaryDirectory() as tmp:
        gz = Path(tmp) / "table.csv.gz"
        csv_path = Path(tmp) / "table.csv"
        _get_to_file(f"/{db}/{tbl}.csv.gz", gz, key, timeout=600)
        with gzip.open(gz, "rb") as src, csv_path.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        df = pd.read_csv(str(csv_path), low_memory=False)
    cache_store(df, table, db_path)
    return df


def _manifest_rows(key: str) -> int | None:
    """Row count the manifest gives for ``key`` (None when unknown or the manifest is not at hand)."""
    m = cached_manifest()
    for d in (m or {}).get("datasets", []):
        if d.get("key") == key:
            return d.get("rows")
    return None


def _check_known(key: str) -> None:
    """A key the manifest does not list is named as unknown (a raw HTTP 404 said nothing)."""
    try:
        m = hosted_manifest()
    except Exception:  # noqa: BLE001 - no key or no network: the download reports that
        return
    keys = {d.get("key") for d in m.get("datasets", [])}
    if keys and key not in keys:
        raise KeyError(
            f"unknown dataset key {key!r}: not a curated table at data.rmorie.com (morie list-datasets shows them)"
        )


def hosted_to_csv(key: str, out: str | Path) -> tuple[int, int]:
    """Write ``db/table`` straight to a CSV file, row by row: (rows, columns).

    For tables too large to build in memory (chicago_crime/incidents, 8.6M rows, was killed
    after its download); nothing is cached.
    """
    import csv

    _check_known(key)
    db, tbl = key.split("/", 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ncol = 0
    with tempfile.TemporaryDirectory() as tmp:
        gz = Path(tmp) / "table.csv.gz"
        _get_to_file(f"/{db}/{tbl}.csv.gz", gz, key, timeout=600)
        with (
            gzip.open(gz, "rt", encoding="utf-8", newline="") as src,
            out.open("w", encoding="utf-8", newline="") as dst,
        ):
            reader, writer = csv.reader(src), csv.writer(dst)
            header = next(reader, [])
            ncol = len(header)
            writer.writerow(header)
            for row in reader:
                writer.writerow(row)
                rows += 1
    return rows, ncol


def hosted_entries(manifest: dict | None) -> list[dict]:
    """Catalog-shaped rows for ``list_datasets()``."""
    out = []
    for d in (manifest or {}).get("datasets", []):
        out.append(
            {
                "key": d["key"],
                "name": (d.get("meta") or {}).get("description") or d.get("source") or d["key"],
                "source": "data.rmorie.com",
                "survey": d.get("source", ""),
                "year": "",
                "type": "hosted",
                "cached": False,
                "rows": d.get("rows"),
                "route": "data.rmorie.com (your MORIE key)",
                "table_name": hosted_table_name(d["key"]),
            }
        )
    return out
