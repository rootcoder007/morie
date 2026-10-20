# SPDX-License-Identifier: AGPL-3.0-or-later
"""HMAC signatures, attestations and signed capsule bundles in
rmoriebricklayer's format (``cpssign()``/``cpsvrf()``, ``cpsatst()``,
``cpsbndl()``/``cpsbndlvrf()``).

morie signs with the ``hmac`` scheme only: HMAC-SHA-256 (RFC 2104) of the
message's UTF-8 bytes under a shared secret, exactly ``chmac(key, msg)``.
The scheme takes no context (``cpssign()`` ignores one for ``hmac``), so an
attestation's ``context`` is ``""`` and its ``prehash`` ``"none"``. Bundles
signed with ML-DSA, SLH-DSA or XMSS keys are verified by
``rmoriebricklayer::cpsbndlvrf()`` and the rmbl-action; :func:`verify_bundle`
checks their file digests and attestation digest and reports their
signature as not verifiable here.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import re

from ._json import pretty
from ._manifest import digest
from ._prov import safe_rel, sha256file

__all__ = [
    "sign",
    "verify_signature",
    "key_id",
    "attest",
    "check_attestation",
    "bundle",
    "read_bundle",
    "verify_bundle",
]

BUNDLE = "capsule_bundle.json"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_FIELDS = ("digest", "scheme", "public", "context", "prehash", "note", "signed_utc")


def _bytes(x, what: str) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8")
    raise TypeError(f"`{what}` must be a string or bytes")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sign(message, key, scheme: str | None = None) -> dict:
    """``{"scheme": "hmac", "signature": <64 hex>}``: HMAC-SHA-256 of ``message`` under ``key``,
    the object ``cpssign(message, key, scheme = "hmac")`` returns. ``scheme`` may only
    be ``"hmac"`` (or ``None``): the post-quantum schemes are signed by the R package."""
    if scheme not in (None, "hmac"):
        raise ValueError(f"morie signs with scheme 'hmac' only; sign {scheme} with rmoriebricklayer::cpssign()")
    k = _bytes(key, "key")
    if not k:
        raise ValueError("`key` must not be empty")
    return {"scheme": "hmac", "signature": hmac.new(k, _bytes(message, "message"), hashlib.sha256).hexdigest()}


def verify_signature(message, sig, key) -> bool:
    """``True`` when ``sig`` (from :func:`sign`) is a valid HMAC of ``message`` under ``key``;
    compared in constant time. A malformed signature is ``False``, as in ``cpsvrf()``."""
    if not isinstance(sig, dict) or sig.get("scheme") != "hmac":
        return False
    tag = sig.get("signature")
    if not isinstance(tag, str) or not _HEX64.match(tag):
        return False
    try:
        want = sign(message, key)["signature"]
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(want, tag.lower())


def key_id(key) -> str:
    """The attestation's ``public`` field for an HMAC key: HMAC-SHA-256 of ``"public"``
    under the key. It names the key (so a verifier can pin it) without revealing it."""
    return sign("public", key)["signature"]


def _payload_digest(payload: dict) -> str:
    return digest({k: payload[k] for k in _FIELDS})


def attest(manifest, key, note: str | None = None) -> dict:
    """An attestation over ``digest(manifest)``, in the field order ``cpsatst()`` writes:
    ``digest``, ``scheme``, ``public``, ``context``, ``prehash``, ``note``,
    ``signed_utc``, ``signature``. The signature is over the SHA-256 (hex) of
    the canonical JSON of the first seven fields."""
    payload = {
        "digest": digest(manifest),
        "scheme": "hmac",
        "public": key_id(key),
        "context": "",
        "prehash": "none",
        "note": "" if note is None else str(note),
        "signed_utc": _now(),
    }
    return {**payload, "signature": sign(_payload_digest(payload), key)}


def check_attestation(att: dict, manifest, key=None, public_key: str | None = None) -> dict:
    """The rows ``cpschkatst()`` reports: ``attestation_complete``, ``manifest_digest``,
    ``key_as_expected`` (with ``public_key``, the expected ``public`` hex) and ``signature``. An ``hmac``
    signature needs ``key``; other schemes are verified by the R package."""
    rows = []

    def note(check, ok, detail=""):
        rows.append({"check": check, "ok": ok is True, "detail": str(detail)})

    att = att if isinstance(att, dict) else {}
    missing = [f for f in (*_FIELDS, "signature") if f not in att]
    note("attestation_complete", not missing, ", ".join(missing))
    try:
        got = digest(manifest)
    except TypeError:
        got = "NA"
    ok = got == att.get("digest")
    note("manifest_digest", ok, got if ok else f"attested {att.get('digest', '(missing)')}, manifest is {got}")
    if public_key is not None:
        kok = str(public_key).lower() == str(att.get("public", "")).lower()
        note(
            "key_as_expected",
            kok,
            "the attestation carries the expected key"
            if kok
            else "the attestation carries a different key from the one expected",
        )
    ok, detail = False, ""
    if not missing:
        if att.get("scheme") != "hmac":
            detail = (
                f"scheme {att.get('scheme')}: verified by rmoriebricklayer::cpsbndlvrf() or the rmbl-action;"
                " morie verifies hmac"
            )
        elif key is None:
            detail = "an hmac signature needs the shared key"
        elif key_id(key) != str(att.get("public", "")).lower():
            detail = "the key given is not the key the attestation names"
        else:
            try:
                ok = verify_signature(_payload_digest(att), att["signature"], key)
            except TypeError:
                ok = False
    note("signature", ok, detail)
    return {"ok": all(r["ok"] for r in rows), "checks": rows}


def _listing(d: str) -> list[str]:
    out = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), d).replace(os.sep, "/"))
    return out


def _inner(b: dict) -> dict:
    return {
        "meta": {"files": b["files"], "manifest_digest": b["manifest_digest"], "bundled_utc": b["bundled_utc"]},
        "results": [],
    }


def bundle(dir, manifest, key, files=None, note: str | None = None, path=None) -> dict:
    """Sign a whole capsule into ``capsule_bundle.json``, as ``cpsbndl()`` does.

    Hashes ``files`` (default: every regular file under ``dir``, dotfiles
    included, except the bundle itself), sorted, records ``digest(manifest)`` and the
    time, and signs the lot with :func:`attest` under the HMAC ``key``. The
    file has the R package's layout (``files``, ``manifest_digest``,
    ``bundled_utc``, ``attestation``) and is written to ``path`` (default
    ``dir/capsule_bundle.json``). Returns the bundle.

    rmoriebricklayer 1.0.0 and the rmbl-action check every file and digest of
    such a bundle but cannot yet check an ``hmac`` attestation: their
    ``attestation:signature`` row fails. Verify the signature with
    :func:`verify_bundle` (or R's ``cpsvrf()`` over the payload digest).
    """
    cpsdir = os.fspath(dir)
    if not os.path.isdir(cpsdir):
        raise NotADirectoryError(f"`dir` does not exist: {cpsdir}")
    path = os.fspath(path) if path is not None else os.path.join(cpsdir, BUNDLE)
    if files is None:
        own = os.path.realpath(path)
        files = [f for f in _listing(cpsdir) if os.path.realpath(os.path.join(cpsdir, f)) != own]
    files = sorted(str(f) for f in files)
    if not files:
        raise ValueError("no files to bundle")
    absent = [f for f in files if not os.path.exists(os.path.join(cpsdir, f))]
    if absent:
        raise FileNotFoundError("these files are not in `dir`: " + ", ".join(absent[:5]))
    entries = []
    for f in files:
        p = os.path.join(cpsdir, f)
        entries.append({"path": f, "sha256": sha256file(p), "size_bytes": os.path.getsize(p)})
    b = {"files": entries, "manifest_digest": digest(manifest), "bundled_utc": _now()}
    b["attestation"] = attest(_inner(b), key, note=note)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(pretty(b) + "\n")
    return b


def read_bundle(path) -> dict:
    """A bundle file read back; ``ValueError`` when it is not one."""
    with open(path, encoding="utf-8") as fh:
        b = json.load(fh)
    if not isinstance(b, dict) or b.get("attestation") is None or b.get("files") is None:
        raise ValueError(f"`{path}` is not a capsule bundle")
    return b


def verify_bundle(bundle, dir, manifest=None, key=None, public_key: str | None = None) -> dict:
    """Check a bundle against a directory, as ``cpsbndlvrf()`` does.

    ``bundle`` is a path or a bundle from :func:`read_bundle`. Rows:
    ``file:<path>`` per listed file (digest), ``no_unlisted_files``, the
    ``attestation:*`` rows of :func:`check_attestation` and, with ``manifest``,
    ``manifest_digest``. Returns ``{"ok", "checks"}``.
    """
    own_rel = BUNDLE
    cpsdir = os.fspath(dir)
    if isinstance(bundle, (str, os.PathLike)):
        bpath = os.path.realpath(os.fspath(bundle))
        bundle = read_bundle(bpath)
        root = os.path.realpath(cpsdir)
        if os.path.commonpath([root, bpath]) == root and bpath != root:
            own_rel = os.path.relpath(bpath, root).replace(os.sep, "/")
    rows = []

    def note(check, ok, detail=""):
        rows.append({"check": check, "ok": ok is True, "detail": str(detail)})

    named = []
    for e in bundle.get("files") or []:
        f = str(e.get("path")) if isinstance(e, dict) else "NA"
        named.append(f)
        p = safe_rel(f, cpsdir)
        if p is None:
            note(f"file:{f}", False, "not a plain relative path inside the directory")
            continue
        if not os.path.exists(p):
            note(f"file:{f}", False, "missing from the directory")
            continue
        got = sha256file(p)
        want = str(e.get("sha256"))
        note(f"file:{f}", got == want, got if got == want else f"expected {want}, got {got}")
    extra = [f for f in _listing(cpsdir) if f not in named and f != own_rel]
    note("no_unlisted_files", not extra, "not covered by the bundle: " + ", ".join(extra[:5]) if extra else "")
    try:
        inner = _inner(bundle)
    except KeyError:
        inner = None
    res = check_attestation(bundle.get("attestation"), inner, key=key, public_key=public_key)
    for r in res["checks"]:
        note("attestation:" + r["check"], r["ok"], r["detail"])
    if manifest is not None:
        got = digest(manifest)
        want = str(bundle.get("manifest_digest"))
        note("manifest_digest", got == want, got if got == want else f"bundle says {want}, manifest is {got}")
    return {"ok": all(r["ok"] for r in rows), "checks": rows}
