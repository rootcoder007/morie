# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-quantum capsule keys and signatures in rmoriebricklayer's objects.

Two families, under the scheme names rmoriebricklayer uses:

* ``"ML-DSA-44"``, ``"ML-DSA-65"``, ``"ML-DSA-87"`` (FIPS 204), the keys of ``fipskg()``:
  ``{"public", "secret", "scheme"}`` (hex), public half ``{"public", "scheme"}``
  (``fipspubkey()``). A signature is ``{"scheme", "signature", "context", "prehash"}``,
  ``context`` the context bytes in base64 (what ``tojs()`` writes for R's raw vector), as
  ``cpssign()`` returns it. Pure ML-DSA, or HashML-DSA with ``prehash``.
* ``"xmss-sha256"`` (RFC 8391), the keys of ``pqckg()``: ``{"root", "pubseed", "skseed",
  "skprf", "height", "next_index", "capacity", "scheme"}``, public half ``{"root", "pubseed",
  "height", "scheme"}`` (``signpubkey()``). A signature is ``{"scheme", "signature", "auth",
  "index", "root", "height", "randomizer", "wire"}``.

A key file is the key object as JSON (what ``tojs(unclass(key), auto_unbox = TRUE)`` writes in
R, and what ``fromjs()`` reads back there). An XMSS key signs ``2^height`` messages and no
more, and never two at one index: :func:`sign` retires ``next_index`` in the used-index record
before the signature exists and advances the key object. The record is rmoriebricklayer's own
file and format (``RMBL_XMSS_INDEX_FILE``, else ``xmss_used_indices.txt`` in
``tools::R_user_dir("rmoriebricklayer", "config")``), so morie and the R package on one machine
share one record, under one lock, and neither can re-sign an index the other used.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import platform
import re
import secrets
import time

from morie.crypto import _mldsa, _xmss

__all__ = ["MLDSA", "XMSS", "SCHEMES", "keygen", "public_key", "public_hex", "read_key", "write_key"]

MLDSA = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")
XMSS = "xmss-sha256"
SCHEMES = (*MLDSA, XMSS)

# FIPS 204 section 5.4: the hash's DER OID (NIST hashAlgs arc 2.16.840.1.101.3.4.2.x) and PH(M)
_OID = b"\x06\x09\x60\x86\x48\x01\x65\x03\x04\x02"
_PREHASH = {
    "sha256": (1, lambda m: hashlib.sha256(m).digest()),
    "sha384": (2, lambda m: hashlib.sha384(m).digest()),
    "sha512": (3, lambda m: hashlib.sha512(m).digest()),
    "sha224": (4, lambda m: hashlib.sha224(m).digest()),
    "sha512_224": (5, lambda m: hashlib.new("sha512_224", m).digest()),
    "sha512_256": (6, lambda m: hashlib.new("sha512_256", m).digest()),
    "sha3_224": (7, lambda m: hashlib.sha3_224(m).digest()),
    "sha3_256": (8, lambda m: hashlib.sha3_256(m).digest()),
    "sha3_384": (9, lambda m: hashlib.sha3_384(m).digest()),
    "sha3_512": (10, lambda m: hashlib.sha3_512(m).digest()),
    "shake128": (11, lambda m: hashlib.shake_128(m).digest(32)),
    "shake256": (12, lambda m: hashlib.shake_256(m).digest(64)),
}
_HEX = re.compile(r"^[0-9a-fA-F]*$")


def _hex(x, n: int | None = None) -> bytes | None:
    if not isinstance(x, str) or len(x) % 2 or not _HEX.match(x):
        return None
    b = bytes.fromhex(x)
    return b if n is None or len(b) == n else None


def _level(scheme: str) -> int:
    return int(scheme.rsplit("-", 1)[1])


def keygen(scheme: str = "ML-DSA-65", height: int = 10) -> dict:
    """A new signing key, as ``fipskg(scheme)`` or (``"xmss-sha256"``) ``pqckg(height)`` makes it.

    A new XMSS key is entered in the used-index record at birth, as ``pqckg()`` enters it, so
    it may sign from index 0.

    >>> k = keygen("ML-DSA-44")
    >>> sorted(k), k["scheme"], len(k["public"]) // 2
    (['public', 'scheme', 'secret'], 'ML-DSA-44', 1312)
    """
    if scheme in MLDSA:
        pk, sk = _mldsa.keygen(_level(scheme))
        return {"public": pk.hex(), "secret": sk.hex(), "scheme": scheme}
    if scheme != XMSS:
        raise ValueError(f"unknown scheme '{scheme}': morie signs with hmac, {', '.join(SCHEMES)}")
    sk, ps, prf = (secrets.token_bytes(32) for _ in range(3))
    root = _xmss.keygen_root(sk, ps, height)
    key = {
        "root": root.hex(),
        "pubseed": ps.hex(),
        "skseed": sk.hex(),
        "skprf": prf.hex(),
        "height": height,
        "next_index": 0,
        "capacity": 1 << height,
        "scheme": XMSS,
    }
    _register(key)
    return key


def public_key(key: dict) -> dict:
    """The public half of a key: ``fipspubkey()`` / ``signpubkey()``."""
    if key.get("scheme") == XMSS:
        return {k: key[k] for k in ("root", "pubseed", "height", "scheme")}
    return {"public": key["public"], "scheme": key["scheme"]}


def public_hex(key: dict) -> str:
    """The attestation's ``public`` field: ML-DSA's public key, XMSS's ``root || pubseed``."""
    if key.get("scheme") == XMSS:
        return (key["root"] + key["pubseed"]).lower()
    return str(key["public"]).lower()


def from_attestation(scheme: str, public: str, height) -> dict | None:
    """The public key an attestation names (``.rmbl_attest_key_from()``); ``None`` if unusable."""
    if scheme == XMSS:
        if not isinstance(public, str) or len(public) != 128 or _hex(public) is None:
            return None
        return {"root": public[:64], "pubseed": public[64:], "height": height, "scheme": XMSS}
    if scheme in MLDSA and _hex(public) is not None:
        return {"public": public, "scheme": scheme}
    return None


def load_key(text) -> dict:
    """A key object from its JSON text (bytes or str); ``ValueError`` when it is not one."""
    try:
        key = json.loads(text)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("the key is not a JSON key object (fipskg() / pqckg() fields)") from None
    if not isinstance(key, dict) or key.get("scheme") not in SCHEMES:
        raise ValueError(f"the key is not a JSON key object of a scheme morie signs with ({', '.join(SCHEMES)})")
    return key


def read_key(path) -> dict:
    """A key object read from a JSON key file."""
    with open(path, "rb") as fh:
        return load_key(fh.read())


def write_key(key: dict, path) -> None:
    """Write a key object as JSON, readable by the owner only, replacing ``path`` atomically
    (an XMSS key's ``next_index`` must never go back)."""
    path = os.fspath(path)
    tmp = f"{path}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(key, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


# ------------------------------------------------------------------ signing


def _mprime(message: bytes, context: bytes, prehash: str) -> bytes:
    if len(context) > 255:
        raise ValueError("`context` must be at most 255 bytes")
    if prehash == "none":
        return bytes([0, len(context)]) + context + message
    if prehash not in _PREHASH:
        raise ValueError(f"unknown prehash '{prehash}': none or one of {', '.join(_PREHASH)}")
    n, ph = _PREHASH[prehash]
    return bytes([1, len(context)]) + context + _OID + bytes([n]) + ph(message)


def sign(message: bytes, key: dict, context: bytes = b"", prehash: str = "none") -> dict:
    """The signature ``cpssign(message, key, ctx = context, prehash = prehash)`` returns.

    ML-DSA signs hedged (32 fresh bytes per signature). An XMSS key signs at its
    ``next_index``, which is retired in the used-index record first and then advanced in
    ``key`` (the object is updated in place: save it, as ``xmssadv()`` is saved in R).
    """
    scheme = key.get("scheme")
    if scheme in MLDSA:
        sk = _hex(key.get("secret"))
        if sk is None:
            raise ValueError("signing needs a key with its secret half")
        if _mldsa.level_of_secret_key(sk) != _level(scheme):
            raise ValueError(f"`secret` is not an {scheme} secret key")
        mp = _mprime(message, context, prehash)
        sig = _mldsa._sign_internal(sk, mp, os.urandom(32))
        return {
            "scheme": scheme,
            "signature": sig.hex(),
            "context": base64.b64encode(context).decode("ascii"),
            "prehash": prehash,
        }
    if scheme != XMSS:
        raise ValueError(f"`key` is not a key of a scheme morie signs with ({', '.join(SCHEMES)})")
    h, idx = key.get("height"), key.get("next_index")
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
        raise ValueError("this key's `next_index` is not a whole number >= 0: the key state is damaged")
    if not isinstance(h, int) or not 1 <= h <= 16:
        raise ValueError("this key's `height` is not a whole number from 1 to 16")
    if idx >= 1 << h:
        raise ValueError(
            f"this key is exhausted: a height-{h} key signs {1 << h} messages and all of them are used."
            " Generate a new key -- reusing an index would break the signature scheme."
        )
    seeds = [_hex(key.get(k), 32) for k in ("skseed", "skprf", "pubseed")]
    if None in seeds:
        raise ValueError("an XMSS signing key needs `skseed`, `skprf` and `pubseed`, 64 hex characters each")
    ids = _ids(key)
    _reserve(ids, idx)  # RFC 8391's order: retire the index before the signature exists
    res = _xmss.sign(seeds[0], seeds[1], seeds[2], h, idx, message)
    if res["root"] != ids["root"]:
        raise ValueError(
            f"this key's `root` does not match its seeds: the key object was edited. No signature is"
            f" returned, and index {idx} stays retired"
        )
    key["next_index"] = idx + 1
    return {
        "scheme": XMSS,
        "signature": res["signature"],
        "auth": res["auth"],
        "index": idx,
        "root": res["root"],
        "height": h,
        "randomizer": res["randomizer"],
        "wire": res["wire"],
    }


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def verify(message: bytes, sig: dict, key: dict, context: bytes = b"", prehash: str | None = None) -> bool:
    """``cpsvrf(message, sig, key, ctx = context, prehash = prehash)``: ``True`` when ``sig`` is
    valid under the public (or signing) ``key``. Anything malformed is ``False``."""
    if not isinstance(sig, dict) or not isinstance(key, dict) or sig.get("scheme") != key.get("scheme"):
        return False
    scheme = key.get("scheme")
    if scheme in MLDSA:
        pk, sg = _hex(key.get("public")), _hex(sig.get("signature"))
        ph = prehash if prehash is not None else (sig.get("prehash") or "none")
        if pk is None or sg is None or not isinstance(ph, str):
            return False
        try:
            if _mldsa.level_of_public_key(pk) != _level(scheme):
                return False
            return _mldsa._verify_internal(pk, _mprime(message, context, ph), sg)
        except ValueError:
            return False
    if scheme != XMSS:
        return False
    one = all(isinstance(sig.get(f), str) and sig.get(f) for f in ("signature", "auth", "randomizer", "root", "wire"))
    if not one or not _whole(sig.get("index")) or not _whole(sig.get("height")):
        return False
    if str(sig["root"]).lower() != str(key.get("root", "")).lower() or sig["height"] != key.get("height"):
        return False
    wire = f"{sig['index']:08x}{sig['randomizer']}{sig['signature']}{sig['auth']}"
    if wire.lower() != sig["wire"].lower():
        return False
    ps, root = _hex(key.get("pubseed"), 32), _hex(key.get("root"), 32)
    if ps is None or root is None:
        return False
    return _xmss.verify(
        ps, root, key["height"], sig["index"], message, sig["signature"], sig["auth"], sig["randomizer"]
    )


# ------------------------------------------------------------------ the used-index record


def record_path() -> str:
    """The XMSS used-index record rmoriebricklayer keeps: ``$RMBL_XMSS_INDEX_FILE``, else
    ``tools::R_user_dir("rmoriebricklayer", "config")/xmss_used_indices.txt``."""
    p = os.environ.get("RMBL_XMSS_INDEX_FILE")
    if p is not None:
        p = os.path.expanduser(p.strip())
        if not p or p.upper() in ("FALSE", "F", "0", "NO", "OFF"):
            raise ValueError(
                "RMBL_XMSS_INDEX_FILE turns the XMSS used-index record off: morie signs with an XMSS key"
                " only with the record on"
            )
        if not os.path.isabs(p):
            raise ValueError(f'RMBL_XMSS_INDEX_FILE is "{p}", a relative path: give an absolute path')
        return p
    base = os.environ.get("R_USER_CONFIG_DIR") or os.environ.get("XDG_CONFIG_HOME")
    if not base:
        if os.name == "nt":
            base = os.path.join(os.environ.get("APPDATA", ""), "R", "config")
        elif platform.system() == "Darwin":
            base = os.path.join(os.path.expanduser("~"), "Library", "Preferences", "org.R-project.R")
        else:
            base = os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "R", "rmoriebricklayer", "xmss_used_indices.txt")


_LINE = re.compile(r"^(k:[0-9a-f]{64}|[0-9a-f]{64}:[0-9a-f]{64})\t(-1|[0-9]{1,9})$")


def _record_read(p: str) -> dict[str, int]:
    rec: dict[str, int] = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
        bad = [i + 1 for i, ln in enumerate(lines) if not _LINE.match(ln)]
        if bad:
            raise ValueError(
                f"{p} (the XMSS used-index record) has {len(bad)} malformed line(s), the first at line {bad[0]}:"
                " a record that cannot be read cannot be trusted, so no XMSS signature is made until it is repaired"
            )
        for ln in lines:
            k, v = ln.split("\t")
            rec[k] = max(int(v), rec.get(k, -1))
    legacy = os.path.join(os.path.dirname(p), "xmss_index.json")  # the record before rmoriebricklayer 0.5.12
    if os.path.exists(legacy):
        try:
            with open(legacy, encoding="utf-8") as fh:
                old = json.load(fh)
            for k, v in old.items():
                if not re.match(r"^[0-9a-fA-F]{64}:[0-9a-fA-F]{64}$", k) or not isinstance(v, int) or v < -1:
                    raise ValueError
                rec[k.lower()] = max(v, rec.get(k.lower(), -1))
        except (ValueError, AttributeError, OSError):
            raise ValueError(f"{legacy} (the XMSS index record written before 0.5.12) cannot be read") from None
    return rec


def _record_write(p: str, rec: dict[str, int]) -> None:
    d = os.path.dirname(p)
    tmp = os.path.join(d, f"{os.path.basename(p)}.xmss-{secrets.token_hex(6)}.tmp")
    body = "".join(f"{k}\t{v}\n" for k, v in rec.items()).encode("utf-8")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.getsize(tmp) != len(body):
            raise OSError("short write")
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    if os.name != "nt":
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


@contextlib.contextmanager
def _locked(p: str):
    """rmoriebricklayer's lock on the record: flock(2) on ``<record>.flock`` (one byte with
    LockFileEx on Windows), waiting at most 10 s."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd = os.open(p + ".flock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            def take():
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

            def give():
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def take():
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def give():
                fcntl.flock(fd, fcntl.LOCK_UN)

        start = time.monotonic()
        while True:
            try:
                take()
                break
            except OSError:
                if time.monotonic() - start >= 10:
                    raise ValueError(f"could not lock the XMSS used-index record {p} within 10 s") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            give()
    finally:
        os.close(fd)


def _ids(key: dict) -> dict:
    sk, ps = _hex(key.get("skseed"), 32), _hex(key.get("pubseed"), 32)
    root = str(key.get("root", "")).lower()
    if sk is None or ps is None or not re.match(r"^[0-9a-f]{64}$", root):
        raise ValueError("this XMSS key's seeds or `root` are not 64 hex characters: do not hand-edit the key")
    fp = hashlib.sha256(("rmoriebricklayer xmss record\n" + key["skseed"].lower()).encode("ascii")).hexdigest()
    return {"id": "k:" + fp, "legacy": f"{root}:{key['pubseed'].lower()}", "root": root}


def _reserve(ids: dict, idx: int) -> None:
    p = record_path()
    with _locked(p):
        rec = _record_read(p)
        known = [rec[k] for k in (ids["id"], ids["legacy"]) if k in rec]
        hi = max(known, default=-1)
        if idx <= hi:
            raise ValueError(
                f"index {idx} of this key was already used: signing it again would break the scheme"
                f" (next index {hi + 1}). Sign with the key file as the last signature left it."
            )
        if not known and idx == 0:
            raise ValueError(
                "this XMSS key is not in the used-index record on this machine, and it claims never to have"
                " signed (next_index 0): it was generated elsewhere or from given seeds, or the record was"
                " deleted. If it has never signed, register it with rmoriebricklayer::xmssregkey(key)"
            )
        rec[ids["id"]] = idx
        _record_write(p, rec)


def _register(key: dict) -> None:
    ids, p = _ids(key), record_path()
    with _locked(p):
        rec = _record_read(p)
        if ids["id"] not in rec:
            rec[ids["id"]] = -1
            _record_write(p, rec)
