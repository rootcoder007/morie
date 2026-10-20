# SPDX-License-Identifier: AGPL-3.0-or-later
"""Results manifests in rmoriebricklayer's format: ``mkmnf()``, ``recrd()``,
``wrmnfjs()``, ``mnfcanon()``, ``mnfdgst()`` and ``mnfrcmp()``."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import sys

from ._json import canon as _canon
from ._json import pretty as _pretty

__all__ = ["manifest", "record", "write_manifest", "read_manifest", "canon", "digest", "compare"]


def _synthetic_flag() -> bool:
    v = os.environ.get("BRICKLAYER_SYNTHETIC", "")
    return v != "" and v.lower() not in ("0", "false", "no", "off")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _env() -> dict:
    from importlib import metadata

    # the distributions behind the loaded top-level modules, as capenv() records loaded namespaces
    dists = metadata.packages_distributions()
    names = {"morie", *(d for m in list(sys.modules) for d in dists.get(m, ()) if "." not in m)}
    pkgs = {}
    for name in sorted(names):
        try:
            pkgs[name] = metadata.version(name)
        except Exception:
            continue
    return {
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "os": f"{platform.system()} {platform.release()}",
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "packages": pkgs,
    }


def manifest(
    meta: dict | None = None, checks: list | None = None, path=None, env: bool = True, canonical: bool = False
) -> dict:
    """A new results manifest, as ``mkmnf()`` makes one, optionally filled and written.

    ``meta`` is the run's metadata (``project``, ``author``, ...). ``synthetic``
    is set to ``True`` when ``meta`` says so or the environment variable
    ``BRICKLAYER_SYNTHETIC`` is set to anything but ``0``/``false``/``no``/``off``,
    and is written only when true. Empty ``meta`` and ``results`` are kept as
    ``[]``, which is how R writes an empty list, so the digest of an empty
    manifest is the one R computes. With ``env`` the Python session is
    recorded under ``environment`` (``python_version``, ``platform``, ``os``,
    ``captured_utc``, ``packages``). ``checks`` is a list of dicts with
    ``name``, ``observed``, ``expected`` and optionally ``tol`` and ``group``,
    each added with :func:`record`; with ``path`` the manifest is written by
    :func:`write_manifest` (``canonical`` as there).
    """
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise TypeError("`meta` must be a dict")
    meta = dict(meta)
    if meta.get("synthetic") is True or _synthetic_flag():
        meta["synthetic"] = True
    m = {"meta": meta or [], "results": []}
    if env:
        m["environment"] = _env()
    for c in checks or []:
        record(m, c["name"], c["observed"], c["expected"], tol=c.get("tol", 0.0001), group=c.get("group", "general"))
    if path is not None:
        write_manifest(m, path, canonical=canonical)
    return m


def record(
    manifest: dict, name: str, observed, expected, tol: float = 0.0001, group: str = "general", synthetic: bool = False
) -> dict:
    """Add one cross-check to ``manifest`` (in place, and returned), as ``recrd()`` does.

    The status is ``INFO`` on synthetic data (``synthetic`` or ``meta.synthetic``),
    ``PASS`` when both values are numbers at most ``tol`` apart, ``DIFFER``
    when further apart, and ``INFO`` otherwise. Recording an existing name
    replaces it.
    """
    if not isinstance(manifest, dict) or "results" not in manifest:
        raise TypeError("`manifest` must be a manifest from manifest()")
    if not isinstance(name, str) or not name:
        raise ValueError("`name` must be one non-empty string")
    if not _num(tol) or not tol >= 0:
        raise ValueError("`tol` must be one number >= 0")
    if isinstance(observed, (list, tuple, dict)) or isinstance(expected, (list, tuple, dict)):
        raise ValueError("`observed` and `expected` must be single values")
    diff = abs(observed - expected) if _num(observed) and _num(expected) else None
    if diff is not None and math.isnan(diff):
        diff = None
    meta = manifest.get("meta")
    syn = bool(synthetic) or (isinstance(meta, dict) and meta.get("synthetic") is True)
    if syn:
        status = "INFO"
    elif diff is not None:
        status = "PASS" if diff <= tol else "DIFFER"
    else:
        status = "INFO"
    if not isinstance(manifest["results"], dict):
        manifest["results"] = {}
    manifest["results"][name] = {
        "group": group,
        "observed": observed,
        "expected": expected,
        "diff": diff,
        "status": status,
        "tol": tol,
        "note": "synthetic data -- comparison not meaningful" if syn else None,
    }
    return manifest


def _flagged(man: dict) -> dict:
    meta = man.get("meta")
    if (isinstance(meta, dict) and meta.get("synthetic") is True) or _synthetic_flag():
        man = dict(man)
        man["meta"] = {**(meta if isinstance(meta, dict) else {}), "synthetic": True}
    return man


def canon(man) -> str:
    """The canonical one-line JSON of a manifest (or any JSON value), as ``mnfcanon()``."""
    return _canon(man)


def digest(man) -> str:
    """SHA-256 of :func:`canon`, as ``mnfdgst()``: 64 lower-case hex characters."""
    return hashlib.sha256(_canon(man).encode("utf-8")).hexdigest()


def write_manifest(manifest: dict, path, canonical: bool = False) -> str:
    """Write ``manifest`` to ``path`` as ``wrmnfjs()`` does: pretty, or with ``canonical`` the
    one line :func:`digest` hashes. A synthetic manifest always says so."""
    if not isinstance(manifest, dict):
        raise TypeError("`manifest` must be a manifest from manifest()")
    man = _flagged(manifest)
    text = _canon(man) if canonical else _pretty(man)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return str(path)


def read_manifest(path) -> dict:
    """A manifest file read back (any JSON object)."""
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if not isinstance(man, dict):
        raise ValueError(f"{path} is not a JSON object")
    return man


def _text(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return f"{v:.15g}"
    return str(v)


def _fmt(x) -> str:
    if x is None:
        return "NA"
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.7g}"


def compare(manifest, data, stats: dict, tol: float | None = None) -> dict:
    """Recompute recorded statistics against ``data``, as ``mnfrcmp()`` does.

    ``manifest`` is a manifest or the path of a manifest file.

    ``stats`` maps result names to functions of ``data``. Each row has
    ``name``, ``recorded``, ``recomputed``, ``delta``, ``tol``, ``status``
    (``MATCH``, ``DIFFER``, ``NOT_RECORDED`` or ``ERROR``) and ``detail``.
    Numbers match when equal (NaN with NaN too) or at most ``tol`` apart
    (else the result's recorded ``tol``, else 0); anything else is compared
    as text. ``unchecked`` lists recorded names not recomputed; ``ok`` needs
    every row ``MATCH`` and nothing unchecked.
    """
    if tol is not None and (not _num(tol) or not tol >= 0):
        raise ValueError("`tol` must be one number >= 0")
    man = manifest if isinstance(manifest, dict) else read_manifest(manifest)
    if man.get("results") is None:
        raise ValueError("`manifest` must be a manifest with recorded results")
    if not isinstance(stats, dict) or not stats or not all(isinstance(k, str) and k for k in stats):
        raise ValueError("`stats` must be a non-empty dict of named functions")
    if not all(callable(f) for f in stats.values()):
        raise ValueError("every element of `stats` must be a function")
    results = man["results"] if isinstance(man["results"], dict) else {}
    rows = []
    for nm, fn in stats.items():
        rec = results.get(nm)
        try:
            got = fn(data)
        except Exception as e:
            rows.append(
                dict(name=nm, recorded=None, recomputed=None, delta=None, tol=None, status="ERROR", detail=str(e))
            )
            continue
        if rec is None:
            rows.append(
                dict(
                    name=nm,
                    recorded=None,
                    recomputed=float(got) if _num(got) else None,
                    delta=None,
                    tol=None,
                    status="NOT_RECORDED",
                    detail="the manifest has no result of this name",
                )
            )
            continue
        use_tol = float(tol) if tol is not None else float(rec.get("tol") or 0)
        ro = rec.get("observed")
        if not _num(ro) or not _num(got):
            same = _text(ro) == _text(got)
            rows.append(
                dict(
                    name=nm,
                    recorded=float(ro) if _num(ro) else None,
                    recomputed=float(got) if _num(got) else None,
                    delta=None,
                    tol=use_tol,
                    status="MATCH" if same else "DIFFER",
                    detail=f"recorded '{_text(ro)}', recomputed '{_text(got)}'",
                )
            )
            continue
        r, g = float(ro), float(got)
        d = abs(r - g)
        same = r == g or (math.isnan(r) and math.isnan(g)) or (not math.isnan(d) and d <= use_tol)
        rows.append(
            dict(
                name=nm,
                recorded=r,
                recomputed=g,
                delta=d,
                tol=use_tol,
                status="MATCH" if same else "DIFFER",
                detail=f"|recorded - recomputed| = {d:.17g}"
                if math.isfinite(d)
                else f"recorded {_fmt(r)}, recomputed {_fmt(g)}",
            )
        )
    unchecked = [n for n in results if n not in stats]
    return {"ok": all(r["status"] == "MATCH" for r in rows) and not unchecked, "results": rows, "unchecked": unchecked}
