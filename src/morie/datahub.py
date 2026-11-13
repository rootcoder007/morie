# SPDX-License-Identifier: AGPL-3.0-or-later
"""Curated datasets at data.rmorie.com, with no sign-in.

The MORIE project materialises public BigQuery datasets (Chicago crime, EPA
air quality, census, FEC, FDA, NOAA, ...) and serves them as CSV from the
edge (Cloudflare R2 behind a Worker), so they are reachable whenever the
internet is. No key is needed and none is sent; a request carries the package
User-Agent the service asks of downloads. ``/manifest.json`` lists every table with its
rows, columns, size, SHA-256 and the BigQuery source it was built from;
``/<db>/<table>.csv.gz`` is the table. Downloads are cached in the dataset
store, so a table is fetched once, not per call.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from morie.fn import _frame_core as pd

DEFAULT_DATA_URL = "https://data.rmorie.com"
_MANIFEST_TTL = 24 * 3600


class DataHubAuthError(RuntimeError):
    """No key, or a key the gateway rejects."""


def data_url() -> str:
    """Base URL of the hosted dataset hub, without a trailing slash.

    ``MORIE_DATA_URL`` overrides; otherwise the signed services document decides
    (``morie.services``). Raises ``RuntimeError`` when the document says the
    service is off.

    Examples:
        >>> data_url().startswith("https://")
        True
    """
    env = os.environ.get("MORIE_DATA_URL", "").strip()
    if env:
        from urllib.parse import urlsplit

        u = urlsplit(env)
        here = u.hostname in ("127.0.0.1", "localhost", "::1")
        if not u.hostname or (u.scheme != "https" and not (u.scheme == "http" and here)):
            raise ValueError(
                f"MORIE_DATA_URL must be an https address (plain http is accepted for this machine only): {env!r}"
            )
        return env.rstrip("/")
    from . import services

    data = services.data()
    if data.get("mode") != "key" or not data.get("base_url"):
        raise RuntimeError(
            "the curated-data service is not available right now "
            f"(see {data.get('request_access') or 'https://rmorie.com/access'})"
        )
    return str(data["base_url"]).rstrip("/")


class _NoAuthRedirect(HTTPRedirectHandler):
    """A redirect never carries an ``Authorization`` header on: urllib copies one to the new
    address, which may be another host. None is sent here; this holds if a caller adds one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            for store in (new.headers, new.unredirected_hdrs):
                for name in [h for h in store if h.lower() == "authorization"]:
                    del store[name]
        return new


_OPENER = build_opener(_NoAuthRedirect)


def urlopen(req, timeout=60):
    """Open ``req`` with the redirect rule above (tests replace this name)."""
    return _OPENER.open(req, timeout=timeout)


_BUSY = (429, 500, 502, 503, 504)


def _wait(exc: HTTPError, attempt: int) -> float:
    """Seconds before asking again: the server's Retry-After (at most a minute), else 1, 2, 4."""
    try:
        asked = float((exc.headers or {}).get("Retry-After", ""))
    except (TypeError, ValueError):
        asked = 0.0
    return min(asked, 60.0) if asked > 0 else float(2**attempt)


def _ask(req, timeout=60):
    """``urlopen`` that asks a busy service again, up to three more times: data.rmorie.com
    answers 429 past its per-address rate (a lab behind one address, ``pull --all``) and 503
    when many clients fetch the same table, both with Retry-After."""
    import time

    for attempt in range(4):
        try:
            return urlopen(req, timeout=timeout)
        except HTTPError as exc:
            if exc.code not in _BUSY or attempt == 3:
                raise
            time.sleep(_wait(exc, attempt))


# what data.rmorie.com does not serve: ICPSR's terms forbid redistribution
_UNSERVED = re.compile(r"^/?(icpsr_|ncvs_38604(/|\.|$))")


def unserved(key: str, entry: dict | None = None) -> str | None:
    """Why ``key`` is not served, or None when it is.

    The ICPSR studies and NCVS 38604 are not redistributed, and neither is a table the manifest
    marks restricted or gives an access other than open or public. Such a table is not listed, and
    asking for it is refused before any request is made (as rmoriebricklayer does).

    Examples:
        >>> unserved("chicago_crime/incidents") is None, "ICPSR" in unserved("icpsr_99999/fake")
        (True, True)
    """
    if _UNSERVED.match(key):
        return (
            f"{key.strip('/')} is not served by data.rmorie.com: ICPSR's terms do not allow "
            "redistribution; download it from ICPSR (https://www.icpsr.umich.edu)"
        )
    e = entry if isinstance(entry, dict) else {}
    meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
    access = e.get("access", meta.get("access"))
    open_access = access is None or (isinstance(access, str) and access.strip().lower() in ("", "open", "public"))
    if e.get("restricted") is True or meta.get("restricted") is True or not open_access:
        src = e.get("source")
        return (
            f"{key.strip('/')} is not served by data.rmorie.com: its source does not allow "
            "redistribution; get it from its source" + (f" ({src})" if isinstance(src, str) and src else "")
        )
    return None


def _refuse_unserved(key: str) -> None:
    """Raise for a table that is not served, from the manifest at hand (no request is made)."""
    try:
        m = cached_manifest() or {}
    except Exception:  # noqa: BLE001 - an unreadable cache: the key's own name still decides
        m = {}
    entry = next((d for d in m.get("datasets", []) if isinstance(d, dict) and d.get("key") == key), None)
    why = unserved(key, entry)
    if why:
        raise DataHubAuthError(why)


def _manifest_cache_path() -> Path:
    from .data import _user_cache_dir

    return _user_cache_dir() / "data_rmorie_manifest.json"


def _headers() -> dict:
    """No sign-in and no key: only the package User-Agent the service asks of downloads."""
    return {"User-Agent": "morie/1 (+https://rmorie.com)"}


def _http_error(exc: HTTPError, path: str) -> Exception:
    if exc.code == 410:
        return DataHubAuthError(
            unserved(path.strip("/"))
            or f"{path.strip('/')} is no longer served by data.rmorie.com: its source does not allow "
            "redistribution; get it from its source"
        )
    if exc.code in (401, 403):
        return DataHubAuthError(_refusal_message(exc.code, path))
    # any other answer: the server's words (no key was sent, so none can be echoed)
    return HTTPError(exc.url, exc.code, str(exc.reason), exc.headers, None)


def _open(path: str, timeout: int = 60):
    why = unserved(path.strip("/"))
    if why:
        raise DataHubAuthError(why)
    req = Request(data_url() + path, headers=_headers())
    try:
        return _ask(req, timeout=timeout)
    except HTTPError as exc:
        # `from None`: the first error's text may carry the key
        raise _http_error(exc, path) from None


def _get(path: str, timeout: int = 60) -> bytes:
    with _open(path, timeout) as resp:
        return resp.read()


def _get_to_file(path: str, dest: Path, label: str, timeout: int = 600) -> int:
    """Stream a gateway file to ``dest`` with progress; a dropped transfer is resumed."""
    from ._progress import download_url

    why = unserved(path.strip("/"))
    if why:
        raise DataHubAuthError(why)
    try:
        return download_url(
            data_url() + path,
            dest,
            label,
            timeout=timeout,
            headers=_headers(),
            opener=lambda req, timeout: _ask(req, timeout=timeout),
        )
    except HTTPError as exc:
        raise _http_error(exc, path) from None


def _refusal_message(code: int, path: str) -> str:
    """What a 401 or 403 means now that the curated tables need no sign-in."""
    why = unserved(path.strip("/"))
    if why:
        return why
    return (
        f"data.rmorie.com refused {path.strip('/')} (HTTP {code}). The curated tables need no "
        "sign-in: update the package (pip install -U morie), then try again."
    )


def _manifest_digest(key: str) -> str | None:
    """The SHA-256 the manifest lists for ``key`` (of the compressed file as served; the field is
    sha256_gz, sha256 or sha), None when it lists none. A field that is not a digest is an error:
    the file cannot be trusted then (the same rule as rmoriebricklayer and rmoriedata)."""
    try:
        m = hosted_manifest()
    except Exception:  # noqa: BLE001 - no manifest: the download itself reports the problem
        m = cached_manifest() or {}
    for d in m.get("datasets", []):
        if d.get("key") != key:
            continue
        for f in ("sha256_gz", "sha256", "sha"):
            if f not in d or d[f] is None:
                continue
            v = d[f]
            v = v[len("sha256:") :] if isinstance(v, str) and v.startswith("sha256:") else v
            if isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v):
                return v.lower()
            raise ValueError(f"the manifest's {f} for {key} is not a SHA-256 hex digest ({v!r})")
    return None


def _manifest_lists(key: str) -> bool:
    """True when a manifest at hand lists ``key``."""
    try:
        m = hosted_manifest()
    except Exception:  # noqa: BLE001 - no manifest now: the cached one, if any
        m = cached_manifest() or {}
    return any(isinstance(d, dict) and d.get("key") == key for d in m.get("datasets", []))


def _check_digest(path: Path, key: str) -> None:
    """The downloaded file against the manifest's SHA-256; a mismatch removes it and stops.

    A table the manifest lists without a digest is loaded with a warning. A table no manifest
    lists (none could be fetched, and none is cached) cannot be checked, and is not loaded.
    """
    want = _manifest_digest(key)
    if want is None:
        if not _manifest_lists(key):
            path.unlink(missing_ok=True)
            raise ValueError(
                f"{key}: no manifest lists this table (data.rmorie.com's manifest could not be read), "
                "so the download cannot be checked; nothing was loaded, try again"
            )
        import warnings

        warnings.warn(
            f"{key}: the manifest lists no SHA-256 for this table, so the download was not checked",
            stacklevel=3,
        )
        return
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        # the table may have been rebuilt since the manifest was cached: read the manifest again, once
        try:
            hosted_manifest(refresh=True)
            want = _manifest_digest(key) or want
        except Exception:  # noqa: BLE001 - no fresh manifest: the mismatch stands
            pass
    if h.hexdigest() != want:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"{key}: the downloaded file does not match the SHA-256 the manifest lists "
            f"(got {h.hexdigest()[:16]}..., expected {want[:16]}...); nothing was loaded, try again"
        )


def offline() -> bool:
    """True under ``MORIE_OFFLINE`` (set by ``morie selftest``): nothing is fetched, the disk copy is used."""
    return os.environ.get("MORIE_OFFLINE", "").strip() not in ("", "0", "false", "no")


def hosted_manifest(refresh: bool = False) -> dict:
    """The gateway's manifest, cached for a day under the user cache dir (only the cached copy when offline)."""
    if offline() and not refresh:
        return cached_manifest() or {}
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
    """Cache-table name for a hosted dataset key: ``hub_`` plus the key with ``/`` written as ``__``.

    Examples:
        >>> hosted_table_name("ethereum_tokens/ethereum_tokens")
        'hub_ethereum_tokens__ethereum_tokens'
    """
    return "hub_" + key.replace("/", "__")


def is_hosted_key(key: str) -> bool:
    """True for a hosted dataset key of the form ``group/table`` (not a file path).

    Examples:
        >>> is_hosted_key("ethereum_tokens/ethereum_tokens"), is_hosted_key("./data.csv"), is_hosted_key("cihi849")
        (True, False, False)
    """
    return "/" in key and not key.startswith(("/", ".")) and not key.endswith("/")


def load_hosted_dataset(key: str, *, db_path: str | Path | None = None, refresh: bool = False):
    """``db/table`` from data.rmorie.com: the dataset store first, then one download."""
    from .data import cache_load, cache_store

    if not is_hosted_key(key):
        raise KeyError(f"{key!r} is not a data.rmorie.com key (expected db/table; see `morie list-datasets`)")
    _refuse_unserved(key)  # before the cache: a copy kept from before is not handed out either
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
        _check_digest(gz, key)
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

    _refuse_unserved(key)
    _check_known(key)
    db, tbl = key.split("/", 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = ncol = 0
    with tempfile.TemporaryDirectory() as tmp:
        gz = Path(tmp) / "table.csv.gz"
        _get_to_file(f"/{db}/{tbl}.csv.gz", gz, key, timeout=600)
        _check_digest(gz, key)
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
        if unserved(d["key"], d):
            continue  # not served, so not offered
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
                "route": "data.rmorie.com (no sign-in)",
                "table_name": hosted_table_name(d["key"]),
            }
        )
    return out
