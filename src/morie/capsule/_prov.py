# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provenance records and capsule verification: what ``vrfcps()`` checks."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re

from . import _csv
from ._json import pretty

__all__ = ["write", "load", "pin", "verify", "sha256file", "check_schema"]

PROVFILE = "data_provenance.json"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_NA = object()  # a field that is present but not one scalar


def sha256file(path) -> str:
    """SHA-256 of a file's bytes, 64 lower-case hex characters (``sha256file()``)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rows(path) -> int:
    return len(_csv.read(path)[1])


def safe_rel(rel, d: str) -> str | None:
    """``d/rel`` when ``rel`` is a plain relative path that stays inside ``d``
    (no ``..``, absolute path, drive letter, ``~`` or symlink out), else ``None``."""
    if isinstance(rel, list) and rel:
        rel = rel[0]
    if not isinstance(rel, str) or not rel:
        return None
    if re.match(r"^([/\\]|~|[A-Za-z]:)", rel):
        return None
    if any(p in ("", ".", "..") for p in rel.replace("\\", "/").split("/")):
        return None
    full = f"{d}/{rel}"  # file.path(): the detail strings name it as R does
    root, f = os.path.realpath(d), os.path.realpath(full)
    try:
        inside = f != root and os.path.commonpath([root, f]) == root
    except ValueError:  # another drive
        inside = False
    return full if inside else None


def _write_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(pretty(obj) + "\n")


def write(
    dir,
    data_file: str,
    url: str | None = None,
    name: str | None = None,
    retrieved: str | None = None,
    provenance_file: str = PROVFILE,
    table: dict | None = None,
    script_file: str | None = None,
) -> dict:
    """Pin a data file of a capsule directory in a provenance record.

    Writes ``provenance_file`` in ``dir`` with ``name`` (default: the data
    file's name without ``.csv``), ``source.url`` (omitted when ``url`` is
    ``None``) and ``source.retrieved`` (default: today, ``YYYY-MM-DD``), and
    ``resource.filename``, ``sha256``, ``size_bytes`` and, for a CSV,
    ``row_count_data_rows`` (data rows as ``read.csv()`` counts them): the
    fields :func:`verify` and ``rmoriebricklayer::vrfcps()`` check. ``table``
    is stored as given; ``script_file``, a file in the capsule, is pinned as
    ``script.filename`` and ``script.sha256``. ``data_file`` and
    ``script_file`` are relative to ``dir`` and must stay inside it. For a
    CSV beside its provenance the bytes are those of the rmbl-action's own
    examples. Returns the record.
    """
    d = os.fspath(dir)
    if not os.path.isdir(d):
        raise NotADirectoryError(f"`dir` is not an existing directory: {d}")
    dpath = safe_rel(data_file, d)
    if dpath is None or not os.path.isfile(dpath):
        raise FileNotFoundError(f"`data_file` must be a file inside {d}: {data_file}")
    if url is not None and (not isinstance(url, str) or not url):
        raise ValueError("`url` must be one non-empty string, or None")
    source = {"url": url} if url is not None else {}
    source["retrieved"] = retrieved or datetime.date.today().isoformat()
    resource = {"filename": data_file, "sha256": sha256file(dpath), "size_bytes": os.path.getsize(dpath)}
    if dpath.lower().endswith(".csv"):
        resource["row_count_data_rows"] = _rows(dpath)
    prov = {
        "name": name if name is not None else re.sub(r"\.csv$", "", os.path.basename(data_file), flags=re.I),
        "source": source,
        "resource": resource,
    }
    if table is not None:
        if not isinstance(table, dict):
            raise TypeError("`table` must be a dict")
        prov["table"] = table
    if script_file is not None:
        sp = safe_rel(script_file, d)
        if sp is None or not os.path.isfile(sp):
            raise FileNotFoundError(f"`script_file` must be a file inside {d}: {script_file}")
        prov["script"] = {"filename": script_file, "sha256": sha256file(sp)}
    ppath = safe_rel(provenance_file, d)
    if ppath is None:
        raise ValueError(f"'{provenance_file}' is not a plain relative path inside the capsule")
    _write_json(prov, ppath)
    return prov


def load(path) -> dict:
    """Read and validate a provenance file; raises ``ValueError`` listing every problem.

    Valid: a JSON object whose ``resource`` has a non-empty ``filename`` and a
    64-hex ``sha256``, with ``size_bytes`` and ``row_count_data_rows`` whole
    numbers >= 0 when present and ``script.sha256`` 64 hex when present -- the
    rules of the rmbl-action's ``capsule.schema.json``.
    """
    with open(path, encoding="utf-8") as fh:
        prov = json.load(fh)
    bad = []
    res = prov.get("resource") if isinstance(prov, dict) else None
    if not isinstance(prov, dict):
        bad.append("not a JSON object")
    elif not isinstance(res, dict):
        bad.append("no `resource` object")
    else:
        if not isinstance(res.get("filename"), str) or not res["filename"]:
            bad.append("resource.filename must be a non-empty string")
        if not isinstance(res.get("sha256"), str) or not _HEX64.match(res["sha256"]):
            bad.append("resource.sha256 must be 64 hex characters")
        for f in ("size_bytes", "row_count_data_rows"):
            v = res.get(f)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                bad.append(f"resource.{f} must be a whole number >= 0")
        scr = prov.get("script")
        if scr is not None and (
            not isinstance(scr, dict) or (scr.get("sha256") is not None and not _HEX64.match(str(scr["sha256"])))
        ):
            bad.append("script.sha256 must be 64 hex characters")
    if bad:
        raise ValueError(f"{path}: " + "; ".join(bad))
    return prov


def pin(dir, provenance_file: str = PROVFILE) -> dict:
    """Re-pin a capsule's provenance to the files as they are now.

    Updates ``resource.sha256``, ``size_bytes`` and ``row_count_data_rows`` (for
    a CSV) and, when a script is named, ``script.sha256``; every other field
    is kept. Returns the updated record.
    """
    cpsdir = os.fspath(dir)
    ppath = safe_rel(provenance_file, cpsdir)
    if ppath is None:
        raise ValueError(f"'{provenance_file}' is not a plain relative path inside the capsule")
    prov = load(ppath)
    res = prov["resource"]
    dpath = safe_rel(res["filename"], cpsdir)
    if dpath is None or not os.path.isfile(dpath):
        raise FileNotFoundError(f"resource.filename is not a file inside the capsule: {res['filename']}")
    res["sha256"] = sha256file(dpath)
    res["size_bytes"] = os.path.getsize(dpath)
    if dpath.lower().endswith(".csv"):
        res["row_count_data_rows"] = _rows(dpath)
    scr = prov.get("script")
    if isinstance(scr, dict) and scr.get("filename"):
        sp = safe_rel(scr["filename"], cpsdir)
        if sp is None or not os.path.isfile(sp):
            raise FileNotFoundError(f"script.filename is not a file inside the capsule: {scr['filename']}")
        scr["sha256"] = sha256file(sp)
    _write_json(prov, ppath)
    return prov


# ------------------------------------------------------------------ schema (valsch())


def _adist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _near(x: str, pool, k: int = 3):
    for p in pool:
        if 0 < _adist(x, p) <= k:
            return p
    return None


_RCLASS = {"logical": "logical", "integer": "integer", "numeric": "numeric", "character": "character"}


def _rfmt(x) -> str:
    return f"{x:.7g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def check_schema(cols: dict, prov: dict) -> list[dict]:
    """The issues ``valsch()`` finds in a CSV's columns (from ``_csv.frame``), each
    ``{"id", "severity", "message"}``; severity ``fatal`` or ``warning``."""
    issues: list[dict] = []

    def add(i, sev, msg):
        issues.append({"id": i, "severity": sev, "message": msg})

    sch = prov.get("schema") if isinstance(prov, dict) else None
    if sch is None:
        return issues
    if not isinstance(sch, (dict, list)):
        raise ValueError("`provenance$schema` must be a list (from infsch())")
    if not sch:
        add(
            "schema_empty",
            "warning",
            "the schema is empty (list()): it lists no columns, types or invariants, so nothing would be checked",
        )
        return issues
    if isinstance(sch, list):
        sch = {}
    known = [
        "expected_columns",
        "expected_types",
        "expected_dtypes",
        "structural_invariants",
        "expected_value_sets",
        "numeric_ranges",
        "max_missing_fraction",
        "synthetic_recipe",
    ]
    for nm in [n for n in sch if n not in known]:
        g = _near(nm, known)
        if g is not None:
            add(
                f"schema_field_{nm}", "fatal", f"unknown schema field '{nm}' (did you mean '{g}'?): it would be ignored"
            )
        else:
            add(f"schema_field_{nm}", "warning", f"schema field '{nm}' is not one valsch() uses: it was not checked")
    inv = sch.get("structural_invariants") if isinstance(sch.get("structural_invariants"), dict) else {}
    inv_known = ["min_data_rows", "max_data_rows"]
    for nm in [n for n in inv if n not in inv_known]:
        g = _near(nm, inv_known, k=1)
        if g is None:
            for k in inv_known:
                diff = [i for i in range(len(nm)) if len(nm) == len(k) and nm[i] != k[i]]
                swapped = len(nm) == len(k) and len(diff) == 2 and diff[1] == diff[0] + 1
                if swapped and nm[diff[0]] == k[diff[1]] and nm[diff[1]] == k[diff[0]]:
                    g = k
                    break
        if g is not None:
            add(
                f"schema_field_{nm}",
                "fatal",
                f"unknown structural invariant '{nm}' (did you mean '{g}'?): it would be ignored",
            )
    types = sch.get("expected_types")
    if types is None:
        types = sch.get("expected_dtypes")
    exp_cols = sch.get("expected_columns")
    if exp_cols is not None:
        exp_cols = exp_cols if isinstance(exp_cols, list) else [exp_cols]
        miss = []
        for c in exp_cols:
            if c not in cols and c not in miss:
                miss.append(c)
        if miss:
            add("missing_columns", "fatal", "Missing required columns: " + ", ".join(map(str, miss)))
    n = len(next(iter(cols.values()))[1]) if cols else 0
    lo, hi = inv.get("min_data_rows"), inv.get("max_data_rows")
    if lo is not None and n < lo:
        add("row_count_low", "warning", f"Row count {n} below expected minimum {lo}")
    if hi is not None and n > hi:
        add("row_count_high", "warning", f"Row count {n} above expected maximum {hi}")
    if isinstance(types, dict):
        num, txt = ("integer", "numeric", "double"), ("character", "factor")
        for col, want in types.items():
            if col not in cols:
                continue
            got = _RCLASS[cols[col][0]]
            if not (got == want or (got in num and want in num) or (got in txt and want in txt)):
                add(f"type_{col}", "warning", f"Column '{col}' is {got}, expected {want}")
    rng = sch.get("numeric_ranges")
    if isinstance(rng, dict):
        for col, r in rng.items():
            if col not in cols or cols[col][0] not in ("integer", "numeric"):
                continue
            ok = [v for v in cols[col][1] if v is not None and v == v]
            if not ok or not isinstance(r, dict):
                continue
            a, b = r.get("min"), r.get("max")
            out = sum(1 for v in ok if (a is not None and v < a) or (b is not None and v > b))
            if out:
                add(f"range_{col}", "warning", f"Column '{col}' has {out} value(s) outside [{_rfmt(a)}, {_rfmt(b)}]")
    mmf = sch.get("max_missing_fraction")
    if isinstance(mmf, dict):
        for col, cap in mmf.items():
            if col not in cols:
                continue
            vals = cols[col][1]
            frac = sum(1 for v in vals if v is None or v != v) / max(1, len(vals))
            if frac > cap:
                add(
                    f"missing_{col}",
                    "warning",
                    f"Column '{col}' is {100 * frac:.1f}% missing, above the expected {100 * cap:.1f}%",
                )
    vs = sch.get("expected_value_sets")
    if isinstance(vs, dict):
        for col, allowed in vs.items():
            if col not in cols:
                continue
            allowed = allowed if isinstance(allowed, list) else [allowed]
            seen, unexpected = set(), []
            for v in cols[col][1]:
                key = "NA" if v is None else v
                if key in seen:
                    continue
                seen.add(key)
                if not any(v == a or (v is not None and str(v) == str(a)) for a in allowed):
                    unexpected.append("NA" if v is None else _rfmt(v) if not isinstance(v, bool) else str(v).upper())
            if unexpected:
                add(f"unexpected_{col}", "warning", f"Column '{col}' has unexpected values: " + ", ".join(unexpected))
    return issues


# ------------------------------------------------------------------ vrfcps()


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 1 and not isinstance(v[0], (list, dict)):
        v = v[0]
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return f"{v:.15g}"
    if isinstance(v, str):
        return v
    return _NA


def _first(v) -> str:
    if isinstance(v, list):
        v = v[0] if v else None
    return "NA" if v is None else str(v)


def _float(v):
    try:
        return float(v[0] if isinstance(v, list) else v)
    except (TypeError, ValueError, IndexError):
        return float("nan")


def verify(dir, provenance_file: str = PROVFILE, data_file=None, manifest_file=None, script_file=None) -> dict:
    """Re-verify a capsule directory, check by check, as ``vrfcps()`` does.

    Returns ``{"ok", "checks", "required", "missing"}``; ``checks`` is a list of
    ``{"check", "ok", "detail"}`` with the R package's names, in its order:
    ``provenance_readable``, ``data_present``, ``data_sha256``,
    ``data_size_bytes`` (when recorded), ``data_readable``, ``data_row_count``
    (when recorded), ``schema_valid`` (when a schema is recorded),
    ``manifest_consistent`` (with a manifest), ``script_sha256`` (with a
    script) and ``data_not_synthetic``. A check that cannot be made is a failed
    row; ``ok`` needs every row to pass and every required check to be made.
    Offline, writes nothing; the only error is a ``cpsdir`` that is not a directory.
    """
    cpsdir = os.fspath(dir)
    provfile, dfile, mnffile, scrfile = provenance_file, data_file, manifest_file, script_file
    if not os.path.isdir(cpsdir):
        raise NotADirectoryError(f"`dir` is not an existing directory: {cpsdir}")
    checks: list[dict] = []

    def note(check, ok, detail=""):
        checks.append({"check": check, "ok": ok is True, "detail": str(detail)})

    ppath = safe_rel(provfile, cpsdir)
    prov = None
    if ppath is not None and os.path.isfile(ppath):
        try:
            with open(ppath, encoding="utf-8") as fh:
                prov = json.load(fh)
        except (OSError, ValueError):
            prov = None
    note(
        "provenance_readable",
        prov is not None,
        f"'{_first(provfile)}' is not a plain relative path inside the capsule" if ppath is None else ppath,
    )
    required = ["provenance_readable", "data_present", "data_sha256", "data_not_synthetic"]
    res = _get(prov, "resource")
    cols = None
    dpath = None
    if dfile is None:
        dfile = _scalar(_get(res, "filename"))
    if dfile is None:
        note("data_present", False, "no data file: the provenance records no resource$filename and none was given")
    elif dfile is _NA:
        note("data_present", False, "resource$filename is not a single string")
    else:
        dpath = safe_rel(dfile, cpsdir)
        if dpath is None:
            note("data_present", False, f"'{_first(dfile)}' is not a plain relative path inside the capsule")
        else:
            present = os.path.isfile(dpath)
            note(
                "data_present",
                present,
                f"{dpath} is a directory, not a data file" if os.path.isdir(dpath) else dpath,
            )
            if not present:
                dpath = None
    synth_sidecar = dpath is not None and os.path.exists(dpath + ".synthetic")
    if dpath is not None:
        pinned = _scalar(_get(res, "sha256"))
        if pinned is None:
            note("data_sha256", False, "no sha256 recorded in the provenance: the data cannot be verified")
        elif pinned is _NA or not _HEX64.match(pinned.strip()):
            note("data_sha256", False, "resource$sha256 is not a single 64-character hex digest")
        else:
            try:
                actual = sha256file(dpath)
            except OSError:
                note("data_sha256", False, "the data file could not be read")
            else:
                expected = pinned.strip().lower()
                ok = actual == expected
                note("data_sha256", ok, actual if ok else f"expected {expected}, got {actual}")
        size = os.path.getsize(dpath)
        if _get(res, "size_bytes") is not None:
            note("data_size_bytes", size == _float(res["size_bytes"]), f"{size} bytes on disk")
        if dpath.lower().endswith(".csv"):
            try:
                cols = _csv.frame(dpath)
            except (OSError, ValueError, UnicodeDecodeError):
                cols = None
            note("data_readable", cols is not None, dpath)
        elif _get(res, "row_count_data_rows") is not None or _get(prov, "schema") is not None:
            note(
                "data_readable",
                False,
                f"{os.path.basename(dpath)} is not a CSV: the recorded row count / schema cannot be checked",
            )
    if cols is not None:
        nrow = len(next(iter(cols.values()))[1]) if cols else 0
        if _get(res, "row_count_data_rows") is not None:
            note("data_row_count", nrow == _float(res["row_count_data_rows"]), f"{nrow} rows on disk")
        if _get(prov, "schema") is not None:
            try:
                issues = check_schema(cols, prov)
                note(
                    "schema_valid",
                    not any(i["severity"] == "fatal" for i in issues),
                    "; ".join(i["message"] for i in issues),
                )
            except ValueError as e:
                note("schema_valid", False, str(e))

    man = None
    if mnffile is None and os.path.exists(os.path.join(cpsdir, "manifest.json")):
        mnffile = "manifest.json"
    if mnffile is not None:
        required.append("manifest_consistent")
        mpath = safe_rel(mnffile, cpsdir)
        if mpath is None or not os.path.exists(mpath):
            note("manifest_consistent", False, f"manifest '{_first(mnffile)}' is missing or outside the capsule")
        else:
            try:
                with open(mpath, encoding="utf-8") as fh:
                    man = json.load(fh)
            except (OSError, ValueError) as e:
                man = None
                note("manifest_consistent", False, f"manifest '{_first(mnffile)}' cannot be read: {e}")
            if man is not None:
                results = _get(man, "results")
                results = results if isinstance(results, dict) else {}
                mismatch = []
                for nm, r in results.items():
                    if not isinstance(r, dict):
                        continue
                    o, e, t, s = r.get("observed"), r.get("expected"), r.get("tol"), r.get("status")
                    num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (o, e))
                    if num and isinstance(t, (int, float)) and s in ("PASS", "DIFFER"):
                        want = "PASS" if abs(o - e) <= t else "DIFFER"
                        if want != s:
                            mismatch.append(nm)
                note(
                    "manifest_consistent",
                    not mismatch,
                    ", ".join(mismatch) if mismatch else f"{len(results)} results re-checked",
                )

    pinned_script = _get(man, "meta", "script_sha256")
    if pinned_script is None:
        pinned_script = _get(prov, "script", "sha256")
    if scrfile is None:
        scrfile = _get(prov, "script", "filename")
    if scrfile is None:
        scrfile = _get(man, "meta", "script_file")
    if scrfile is not None or pinned_script is not None:
        required.append("script_sha256")
        spath = safe_rel(scrfile, cpsdir) if scrfile is not None else None
        if spath is None or not os.path.exists(spath):
            note(
                "script_sha256",
                False,
                "a script digest is pinned but no script file is named (script_file)"
                if scrfile is None
                else f"script '{_first(scrfile)}' is missing or outside the capsule",
            )
        elif pinned_script is None:
            note("script_sha256", False, "no script digest is pinned in the manifest or the provenance")
        else:
            actual = sha256file(spath)
            note("script_sha256", actual == _first(pinned_script), actual)

    synth_manifest = _get(man, "meta", "synthetic") is True
    synth_prov = _get(prov, "synthetic") is True or _get(res, "synthetic") is True
    if synth_sidecar:
        detail = f"{os.path.basename(dpath)}.synthetic is present: these data were generated, not fetched"
    elif synth_manifest:
        detail = "the manifest records synthetic = true"
    elif synth_prov:
        detail = "the provenance records synthetic = true"
    elif dpath is None:
        detail = "no data file to check"
    else:
        detail = "no synthetic marker: sidecar, manifest and provenance all say real data"
    note("data_not_synthetic", not (synth_sidecar or synth_manifest or synth_prov), detail)

    names = [c["check"] for c in checks]
    missing = [r for r in required if r not in names]
    return {
        "ok": all(c["ok"] for c in checks) and not missing,
        "checks": checks,
        "required": required,
        "missing": missing,
    }
