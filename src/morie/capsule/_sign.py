# SPDX-License-Identifier: AGPL-3.0-or-later
"""Signatures, attestations and signed capsule bundles in rmoriebricklayer's
format (``cpssign()``/``cpsvrf()``, ``cpsatst()``, ``cpsbndl()``/``cpsbndlvrf()``).

Three kinds of key, as in the R package:

* a shared secret (a string or bytes): scheme ``hmac``, HMAC-SHA-256 (RFC 2104)
  of the message's UTF-8 bytes, exactly ``chmac(key, msg)``. It takes no
  context, so an attestation's ``context`` is ``""`` and its ``prehash``
  ``"none"``;
* an ML-DSA key (FIPS 204) from :func:`keygen` or ``fipskg()``: schemes
  ``ML-DSA-44``, ``ML-DSA-65``, ``ML-DSA-87``, with an optional context
  string and pre-hash;
* an XMSS key (RFC 8391) from :func:`keygen` or ``pqckg()``: scheme
  ``xmss-sha256``, ``2^height`` signatures and never two at one index.

The key objects, signatures and the used-index record are the R package's
own (see :mod:`morie.capsule._pq`): a bundle signed here verifies with
``cpsbndlvrf()`` and the rmbl-action, and one signed there verifies here.
Bundles signed with SLH-DSA keys are verified by the R package; here their
signature row fails with that said.
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
import re

from . import _pq
from ._json import pretty
from ._manifest import digest
from ._prov import safe_rel, sha256file

__all__ = [
    "keygen",
    "public_key",
    "read_key",
    "write_key",
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


def keygen(scheme: str = "ML-DSA-65", height: int = 10) -> dict:
    """A new post-quantum signing key, as ``fipskg(scheme)`` or, for ``"xmss-sha256"``,
    ``pqckg(height)`` returns it (a dict with the R object's fields). Save it with
    :func:`write_key`; give others :func:`public_key`.

    >>> k = keygen("ML-DSA-44")
    >>> s = sign("a manifest digest", k)
    >>> s["scheme"], verify_signature("a manifest digest", s, public_key(k))
    ('ML-DSA-44', True)
    """
    return _pq.keygen(scheme, height)


public_key = _pq.public_key
read_key = _pq.read_key
write_key = _pq.write_key


def sign(message, key, scheme: str | None = None, context=None, prehash: str = "none") -> dict:
    """The object ``cpssign(message, key, ...)`` returns.

    With a shared secret (a string or bytes), ``{"scheme": "hmac", "signature": <64 hex>}``:
    HMAC-SHA-256 of ``message`` under ``key``. With a key from :func:`keygen` (or a key file of
    ``fipskg()`` / ``pqckg()``), its scheme's signature; ``context`` (a string or bytes, at most
    255 bytes) and ``prehash`` (``"none"``, ``"sha256"``, ``"sha512"``, ...) apply to ML-DSA.
    An XMSS key's ``next_index`` is advanced in place: save the key after signing.
    ``scheme``, when given, must be the key's.

    >>> sign("what do ya want for nothing?", "Jefe")["signature"][:16]
    '5bdcc146bf60754e'
    """
    if isinstance(key, dict):
        if scheme not in (None, key.get("scheme")):
            raise ValueError(f"`scheme = {scheme!r}` does not match the key's scheme {key.get('scheme')!r}")
        ctx = b"" if context is None else _bytes(context, "context")
        return _pq.sign(_bytes(message, "message"), key, ctx, prehash)
    if scheme not in (None, "hmac"):
        raise ValueError(f"scheme {scheme} needs a key from keygen() (or fipskg() / pqckg()), not a shared secret")
    k = _bytes(key, "key")
    if not k:
        raise ValueError("`key` must not be empty")
    return {"scheme": "hmac", "signature": hmac.new(k, _bytes(message, "message"), hashlib.sha256).hexdigest()}


def verify_signature(message, sig, key, context=None, prehash: str | None = None) -> bool:
    """``True`` when ``sig`` (from :func:`sign` or ``cpssign()``) is valid for ``message`` under
    ``key``: the shared secret for ``hmac`` (compared in constant time), else the public (or
    signing) key, with the ``context`` it was signed with. A malformed signature is ``False``,
    as in ``cpsvrf()``."""
    if isinstance(key, dict):
        try:
            ctx = b"" if context is None else _bytes(context, "context")
            return _pq.verify(_bytes(message, "message"), sig, key, ctx, prehash)
        except TypeError:
            return False
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
    """The attestation's ``public`` field. For an HMAC key, HMAC-SHA-256 of ``"public"`` under
    the key: it names the key (so a verifier can pin it) without revealing it. For an ML-DSA
    key its public key, for an XMSS key ``root || pubseed``."""
    if isinstance(key, dict):
        return _pq.public_hex(key)
    return sign("public", key)["signature"]


def _payload_digest(payload: dict) -> str:
    return digest({k: payload[k] for k in _FIELDS})


def attest(manifest, key, note: str | None = None, context: str | None = None, prehash: str = "none") -> dict:
    """An attestation over ``digest(manifest)``, as ``cpsatst()`` writes it, in its field order:
    ``digest``, ``scheme``, ``public``, ``context``, ``prehash``, ``note``,
    ``signed_utc``, ``signature``. The signature is over the SHA-256 (hex) of
    the canonical JSON of the first seven fields. ``context`` and ``prehash`` are bound into
    an ML-DSA signature (XMSS records them and its signature does not use them); an hmac
    attestation takes neither."""
    scheme = key.get("scheme") if isinstance(key, dict) else "hmac"
    if scheme == "hmac" and (context is not None or prehash != "none"):
        raise ValueError("an hmac attestation takes no `context` and no `prehash`")
    if context is not None and not isinstance(context, str):
        raise TypeError("`context` must be a string")
    payload = {
        "digest": digest(manifest),
        "scheme": scheme,
        "public": key_id(key),
        "context": "" if context is None else context,
        "prehash": prehash,
        "note": "" if note is None else str(note),
        "signed_utc": _now(),
    }
    signed = _payload_digest(payload)
    # XMSS records a context and pre-hash and its signature does not use them, as in cpsatst()
    sig = sign(signed, key, context=context, prehash=prehash) if scheme in _pq.MLDSA else sign(signed, key)
    return {**payload, "signature": sig}


def check_attestation(att: dict, manifest, key=None, public_key=None) -> dict:
    """The rows ``cpschkatst()`` reports: ``attestation_complete``, ``manifest_digest``,
    ``key_as_expected`` (with ``public_key``: the expected ``public`` hex, or a key object) and
    ``signature``. An ``hmac`` signature needs the shared ``key``; an ML-DSA or XMSS one is
    verified under the public key the attestation carries (pin it with ``public_key``)."""
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
        want = _pq.public_hex(public_key) if isinstance(public_key, dict) else str(public_key)
        kok = want.lower() == str(att.get("public", "")).lower()
        note(
            "key_as_expected",
            kok,
            "the attestation carries the expected key"
            if kok
            else "the attestation carries a different key from the one expected",
        )
    ok, detail = False, ""
    if not missing and att.get("scheme") in _pq.SCHEMES:
        sig = att["signature"]
        pub = _pq.from_attestation(att["scheme"], att["public"], sig.get("height") if isinstance(sig, dict) else None)
        if pub is None:
            detail = "the attested public key is not usable for this scheme"
        elif att["scheme"] == _pq.XMSS:
            ok = verify_signature(_payload_digest(att), sig, pub)
        else:
            ctx = att["context"] if isinstance(att["context"], str) and att["context"] else None
            ok = verify_signature(_payload_digest(att), sig, pub, context=ctx, prehash=str(att["prehash"]))
    elif not missing:
        if att.get("scheme") != "hmac":
            detail = (
                f"scheme {att.get('scheme')}: verified by rmoriebricklayer::cpsbndlvrf() or the rmbl-action;"
                " morie verifies hmac, ML-DSA and XMSS"
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


def bundle(
    dir,
    manifest,
    key,
    files=None,
    note: str | None = None,
    path=None,
    context: str | None = None,
    prehash: str = "none",
) -> dict:
    """Sign a whole capsule into ``capsule_bundle.json``, as ``cpsbndl()`` does.

    Hashes ``files`` (default: every regular file under ``dir``, dotfiles
    included, except the bundle itself), sorted, records ``digest(manifest)`` and the
    time, and signs the lot with :func:`attest` under ``key`` (a shared secret, or an
    ML-DSA or XMSS key from :func:`keygen`; ``context`` and ``prehash`` as there). The
    file has the R package's layout (``files``, ``manifest_digest``,
    ``bundled_utc``, ``attestation``) and is written to ``path`` (default
    ``dir/capsule_bundle.json``). Returns the bundle. An XMSS key is advanced in place:
    save it (:func:`write_key`) before signing again.

    ``rmoriebricklayer::cpsbndlvrf()`` verifies such a bundle: an ML-DSA or XMSS one
    under the public key it carries (``keyexp =`` pins it), an ``hmac`` one with the
    shared key (``key =``, rmoriebricklayer >= 1.0.0 (srr)). :func:`verify_bundle`
    checks it here.

    >>> import tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (d / "d.csv").write_text("x\\n1\\n")
    >>> k = keygen("ML-DSA-44")
    >>> b = bundle(d, {"meta": {}, "results": []}, k)
    >>> b["attestation"]["scheme"], verify_bundle(str(d / BUNDLE), d, public_key=public_key(k))["ok"]
    ('ML-DSA-44', True)
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
    b["attestation"] = attest(_inner(b), key, note=note, context=context, prehash=prehash)
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


def verify_bundle(bundle, dir, manifest=None, key=None, public_key=None) -> dict:
    """Check a bundle against a directory, as ``cpsbndlvrf()`` does.

    ``bundle`` is a path or a bundle from :func:`read_bundle`. Rows:
    ``file:<path>`` per listed file (digest), ``no_unlisted_files``, the
    ``attestation:*`` rows of :func:`check_attestation` and, with ``manifest``,
    ``manifest_digest``. ``key`` is the shared secret of an ``hmac`` bundle; ``public_key``
    (hex or a key object) pins the key any bundle must be signed with. Returns
    ``{"ok", "checks"}``.
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
