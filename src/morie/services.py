"""The signed MORIE services document: where the hosted services are, and whether they are open.

The project publishes one document at ``https://rmorie.com/.well-known/morie-services.json``
with a detached ML-DSA-44 signature beside it. The packages pin the public key and the
URL and nothing else, so an endpoint, a model list or an access mode can change on the
site without a package release, and a hijacked site cannot redirect anyone's key: the
signature fails and the package keeps its last good copy.

Resolution order: a fresh cached copy; the live document (fetched, verified, newer than
the cache); the stale cached copy (verified); the copy bundled with the package; and
finally a document with every mode ``"off"``.

The hosted tier is a last resort behind a local model or the user's own API key; keys are
personal and issued on request at https://rmorie.com/access.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .crypto._dilithium import mldsa_verify

SERVICES_URL = "https://rmorie.com/.well-known/morie-services.json"
CONTEXT = b"morie-services"
# ML-DSA-44 public key of the document signer (generated 2026-10-06). Rotating it is a
# package release by design; rotating anything the document names is not.
PUBKEY_HEX = (
    "ed5aa38192b6ad40e692856549fa2d8977f551d6cfe36c80c5235c19d0762d3a"
    "f4f82b0032cc765d1bee192414d862795b701b6614ea5d5e0bbe3189d0dc656b"
    "f65efc2a2bcc660fb1e90ef66f0349554f2fc14a6e25331bf06116d76ef07a6c"
    "b8b03691ae384f57a1d0ff999551479c22174ae2de427ad896d2729cdecc69e6"
    "e3e97c0e2971aebce530ea7c944dfc15525b431dde0fa3d45c6881cdf1d8dc1f"
    "80cb30195de49d84b4d4fe1c83540917ae95ba315dc05ac0526ca5f21b98d263"
    "af8c6249bc9a97d8d32c0d833a99df87bddc3e8d0ec15a79e0b0f41cadb6256c"
    "fbc94e2458d6b1a7fb4979b1884e5aaa9bd86969b02ac02a98ceae757cea462c"
    "869f1d8a9c264be702db980fa91192949d6f2de7aff823f06aa752443c1703e3"
    "230917adeef7283c2be5ed2b2bfc38957cac9398a44111a7e4376bfce838b51e"
    "285f6ee5a38ac37c5053f3d8b67f08d3f1299c8dabb0cbd588d6176bf08507cd"
    "1299e02abec1fb72c954993b31dbe7a0d7e0a67f78117ee315e9c4ed48178f93"
    "09816fb7e6fd2c3c5d88dcfaee6f4ae5acd9c19f96a841e9b19d80dfb47c424a"
    "0216bc2532767af9f1906ad02fbf61eb957eb899fb83c2486c6aead0268d5ae4"
    "48c46afdb8c58f506b93b2d1ca1124d3346f7995f10ab371bfdb840f6ae31a52"
    "d0121133cc53ba3db24f9a4485fb2e2245cf4d7ee8b1eb7c120973e6df3c2f1f"
    "3828543d541a3ead9630199063921c8e185f90609d3f220e5b5ab616cb8537e2"
    "708cfd319e8bebecda5c32eb832a8921f3f81c266922f1709ef4d34613d39786"
    "444c9dc1defdb7540a912295b58adb8bdb67852006ecbb2033c59a2594aedd83"
    "4a40a17d584e703ffb0f3ffac13b39b3c682abaf48fe6f17591fe51ce095e9dd"
    "dbc3b3a48d413b9a6b950f7b5c2b394a131c0fe494feb11c7f870ebce59f83b6"
    "8c48a86ea83488203c8110d1b8b0c4ad4a8c168fbb584cd9852d04e536093dc6"
    "d693e86bba31e2dcc7af667d382b80c5bbb4bf02b721b9a903e017fbb0c13d67"
    "d42fd31a3673e1709271d5514fb59773d3f8ad171d78b5824da3496665f26147"
    "f9fae8a474e5574c0bf5ee5247dd81cf4d5a5b1c44e71ada02a31dd9cb85e645"
    "330a3db8c40acbc0135efb5630113595c446fd8bf82e4c382aa4686247d8e5e1"
    "05fe8ace027595cd330fde4e7c8f68f42003aed94952d945a87fa658798c2e44"
    "a182f024d3b074a2e90ce5d0050a454a17c3fcfde995eb272e61a8994f6c74dd"
    "60784d02381e9115b9c3dd02a79ff4e02385ced5b2ebba324855cf1b50c384aa"
    "fe6e6fbaf656e326cd9522bc7102bf95c96cb338f7a62da9a3a97cb1b348c853"
    "75bd2afcb915ccb238939229967e2b839c99468bfe7f44219bf6d0b9680dff0a"
    "3f849ceef5954e9c5836c424d7405e16711af9b9b6361257ac1c5303fa92f5d3"
    "178aeb4aa7a28e47ca86ee388b101d8b62f08e8bbe9cd104251ff324382d5018"
    "a67064ec1de0786da7c61a20d3ade9e1cb0c0f44b11627eeee1d9b10b508c459"
    "6c00214f675d046548c42d5e64e567d42e6836195bd215af78468af1ee4ba3bb"
    "6c5d75f0bafaf45b5c863497b6fb4ecbda5e46878a124fbab75ea6f773c8bf58"
    "7843701ef1c0d8fa41abeef91108e540af348558e069d33a125df7d4ec800a53"
    "e1ec04ce0c8d84355c4031598e3ebd18e0b6da42be5e4adbae3bbf5041b4a26f"
    "28c2cc5f9614c6a9dbbb16ad72288f95faaf9cbe2fa9471343c3e78c8e5ac7c7"
    "b3036fa3a8ececbeb1f33c65c5f0cec3be6a85bc3b5691724686ccacdaf9cb0e"
    "0d7b6c0119858a41a5adabfcae656263c835b08d671553ea0aa76034bd9c4f7e"
)
ACCESS_URL = "https://rmorie.com/access"
_memo: dict | None = None


def _cache_path() -> Path:
    from .data import _user_cache_dir

    return _user_cache_dir() / "morie-services.json"


def _bundled_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "morie-services.json"


def off_document() -> dict:
    """The document every field falls back to: nothing hosted is reachable."""
    return {
        "version": 1,
        "issued": "1970-01-01T00:00:00Z",
        "notice": "The hosted MORIE services could not be verified from this machine. "
        "Local models and your own API keys keep working.",
        "llm": {
            "mode": "off",
            "base_url": "",
            "auth_url": "",
            "default_model": "",
            "models": [],
            "request_access": ACCESS_URL,
        },
        "data": {
            "mode": "off",
            "base_url": "",
            "license": "https://rmorie.com/data-license",
            "request_access": ACCESS_URL,
        },
    }


def verify(doc_bytes: bytes, sig_json: str, pubkey_hex: str | None = None) -> bool:
    """True when ``sig_json`` is a valid detached ML-DSA-44 signature over exactly ``doc_bytes``."""
    try:
        sig = json.loads(sig_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(sig, dict) or sig.get("scheme") != "ML-DSA-44" or sig.get("context") != CONTEXT.decode():
        return False
    signature = sig.get("signature")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-fA-F]+", signature or ""):
        return False
    try:
        return bool(
            mldsa_verify(doc_bytes, bytes.fromhex(signature), bytes.fromhex(pubkey_hex or PUBKEY_HEX), context=CONTEXT)
        )
    except (TypeError, ValueError):
        return False


def _time(s: object) -> float | None:
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _url_ok(u: object) -> bool:
    if not isinstance(u, str):
        return False
    if u == "":
        return True
    return bool(re.fullmatch(r"https://[A-Za-z0-9.-]+(:[0-9]+)?(/[^\s@]*)?", u)) and "@" not in u


def parse(doc_bytes: bytes) -> dict | None:
    """The parsed v1 document, shape-checked, or None when it is not one."""
    try:
        d = json.loads(doc_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("version") != 1:
        return None
    llm, data = d.get("llm"), d.get("data")
    if not isinstance(llm, dict) or not isinstance(data, dict):
        return None
    if llm.get("mode") not in ("off", "key") or data.get("mode") not in ("off", "key"):
        return None
    if _time(d.get("issued")) is None:
        return None
    if (
        not _url_ok(llm.get("base_url", ""))
        or not _url_ok(llm.get("auth_url", ""))
        or not _url_ok(data.get("base_url", ""))
    ):
        return None
    models = llm.get("models") or []
    llm["models"] = [str(m) for m in models] if isinstance(models, list) else []
    llm["default_model"] = str(llm.get("default_model") or "")
    d["notice"] = str(d.get("notice") or "")
    return d


def _read(path: Path, pubkey_hex: str | None = None) -> dict | None:
    sig_path = path.with_suffix(".sig")
    if not path.is_file() or not sig_path.is_file():
        return None
    doc_bytes = path.read_bytes()
    if not verify(doc_bytes, sig_path.read_text(encoding="utf-8"), pubkey_hex):
        return None
    return parse(doc_bytes)


def _write(path: Path, doc_bytes: bytes, sig_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(doc_bytes)
    path.with_suffix(".sig").write_text(sig_text, encoding="utf-8")


def _download(url: str, timeout: float) -> tuple[int, bytes]:
    """One seam for the HTTP layer: (status, body). No archive fallback here: an archived copy
    of this document is exactly the stale endpoint list the signature and the issue date exist
    to refuse."""
    resp = httpx.get(url, timeout=timeout, follow_redirects=False)
    return resp.status_code, resp.content


def _sig_url(url: str) -> str:
    """The detached signature beside a document URL: ``.json`` becomes ``.sig``; a mirror
    URL without the suffix gets ``.sig`` appended (it used to fetch the document as its own
    signature and fail closed)."""
    return url[:-5] + ".sig" if url.endswith(".json") else url + ".sig"


def fetch(timeout: float = 20.0) -> tuple[dict, bytes, str] | None:
    """The live document, verified and parsed: (document, bytes, signature text), or None."""
    url = os.environ.get("MORIE_SERVICES_URL", "").strip() or SERVICES_URL
    if not _url_ok(url):
        return None
    try:
        status, doc_bytes = _download(url, timeout)
        if status != 200 or not doc_bytes:
            return None
        status, sig_bytes = _download(_sig_url(url), timeout)
        if status != 200 or not sig_bytes:
            return None
        sig_text = sig_bytes.decode("utf-8")
    except (httpx.HTTPError, UnicodeDecodeError, OSError):
        return None
    if not verify(doc_bytes, sig_text):
        return None
    doc = parse(doc_bytes)
    return (doc, doc_bytes, sig_text) if doc else None


def forget() -> None:
    """Drop the per-process memo (tests; a fresh ``services()`` call re-reads)."""
    global _memo
    _memo = None


def services(refresh: bool = False, max_age: float = 86400, timeout: float = 20.0, offline: bool = False) -> dict:
    """The hosted MORIE services as the project site currently describes them.

    Returns the parsed document with an extra ``"_source"`` key: ``"live"``, ``"cache"``,
    ``"bundled"`` or ``"off"``. ``offline=True`` never fetches (the cached copy, then the
    bundled one); ``refresh=True`` fetches even when the cache is fresh.

    >>> s = services(offline=True)
    >>> s["llm"]["mode"] in ("key", "off") and s["_source"] in ("cache", "bundled", "off")
    True
    """
    global _memo
    if offline and not refresh and _memo is not None:
        return _memo
    out = _resolve(refresh, max_age, timeout, offline)
    _memo = out
    return out


def _resolve(refresh: bool, max_age: float, timeout: float, offline: bool) -> dict:
    cache = _cache_path()
    bundled = _read(_bundled_path())
    cached = _read(cache)
    # The floor every accepted document must reach is the copy shipped with the
    # package: a validly signed but OLDER document (any one ever published) in the
    # cache would otherwise pin a retired endpoint, and it is the endpoint that
    # receives the user's key. A cache earns its place only by being NEWER than the
    # bundled copy; the same date adds nothing, an older one is deleted.
    if cached is not None and bundled is not None and (_time(cached["issued"]) or 0) <= (_time(bundled["issued"]) or 0):
        if (_time(cached["issued"]) or 0) < (_time(bundled["issued"]) or 0):
            with contextlib.suppress(OSError):
                cache.unlink()
                cache.with_suffix(".sig").unlink()
        cached = None
    floor = max([(_time(d["issued"]) or 0) for d in (cached, bundled) if d is not None], default=0)
    fresh = cached is not None and not refresh and (time.time() - cache.stat().st_mtime) < max_age
    if cached is not None and (fresh or offline):
        return {**cached, "_source": "cache"}
    live = None if offline else fetch(timeout)
    if live is not None:
        doc, doc_bytes, sig_text = live
        # a document older than the one already held, or than the bundled one, is a rollback
        if (_time(doc["issued"]) or 0) >= floor:
            with contextlib.suppress(OSError):
                _write(cache, doc_bytes, sig_text)
            return {**doc, "_source": "live"}
    if cached is not None:
        return {**cached, "_source": "cache"}
    if bundled is not None:
        return {**bundled, "_source": "bundled"}
    return {**off_document(), "_source": "off"}


def llm() -> dict:
    """The ``llm`` block of the current document (offline: cache, then bundled)."""
    return services(offline=True)["llm"]


def data() -> dict:
    """The ``data`` block of the current document (offline: cache, then bundled)."""
    return services(offline=True)["data"]


def access_hint() -> str:
    """One line on how to get a hosted key."""
    return f"keys are personal and issued on request at {llm().get('request_access') or ACCESS_URL}; store one with `morie login --token`"
