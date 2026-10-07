"""The signed services document: verification, caching, rollback refusal, fallbacks, wiring."""

from __future__ import annotations

import json
import time

import pytest

from morie import services
from morie.crypto._dilithium import mldsa_keygen, mldsa_sign

PK, SK = mldsa_keygen(44, seed=bytes(32))


def doc(
    issued="2026-10-06T12:00:00Z",
    llm_mode="key",
    data_mode="key",
    base="https://gw.example.org",
    version=1,
    data_base="https://tables.example.org",
):
    return json.dumps(
        {
            "version": version,
            "issued": issued,
            "notice": "test notice",
            "llm": {
                "mode": llm_mode,
                "base_url": base,
                "auth_url": (base + "/auth") if base else "",
                "default_model": "test-model:cloud",
                "models": ["test-model:cloud", "other:cloud"],
                "request_access": "https://rmorie.com/access",
            },
            "data": {
                "mode": data_mode,
                "base_url": data_base,
                "license": "https://rmorie.com/data-license",
                "request_access": "https://rmorie.com/access",
            },
        }
    ).encode()


def sign(doc_bytes, sk=SK, context=b"morie-services", scheme="ML-DSA-44"):
    sig = mldsa_sign(doc_bytes, sk, context=context, deterministic=True)
    return json.dumps({"scheme": scheme, "context": context.decode(), "signature": sig.hex()})


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """A test key, an empty cache directory, no mirror override, no memo."""
    monkeypatch.setattr(services, "PUBKEY_HEX", PK.hex())
    monkeypatch.setattr(services, "_cache_path", lambda: tmp_path / "cache" / "morie-services.json")
    monkeypatch.delenv("MORIE_SERVICES_URL", raising=False)
    for v in ("MORIE_HOSTED_BASE_URL", "MORIE_HOSTED_AUTH_URL", "MORIE_HOSTED_MODEL", "MORIE_DATA_URL"):
        monkeypatch.delenv(v, raising=False)
    services.forget()
    yield tmp_path
    services.forget()


def serve(monkeypatch, pairs: dict):
    def fake(url, timeout):
        if url not in pairs:
            return 404, b""
        body = pairs[url]
        return 200, body if isinstance(body, bytes) else body.encode()

    monkeypatch.setattr(services, "_download", fake)


SIG_URL = services.SERVICES_URL[: -len(".json")] + ".sig"


def test_bundled_document_verifies_with_the_pinned_key(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "_cache_path", lambda: tmp_path / "none" / "morie-services.json")
    services.forget()
    s = services.services(offline=True)
    assert s["_source"] == "bundled"
    assert s["llm"]["mode"] == "key" and s["llm"]["base_url"] == "https://llm.rmorie.com"
    assert s["llm"]["auth_url"] == "https://llm.rmorie.com/auth"
    assert s["llm"]["default_model"] in s["llm"]["models"]
    assert s["data"]["mode"] == "key" and s["data"]["base_url"] == "https://data.rmorie.com"
    assert "last resort" in s["notice"]
    p = services._bundled_path()
    b = bytearray(p.read_bytes())
    assert services.verify(bytes(b), p.with_suffix(".sig").read_text())
    b[-2] ^= 0x20
    assert not services.verify(bytes(b), p.with_suffix(".sig").read_text())
    services.forget()


def test_live_document_is_verified_cached_then_served_from_cache(svc, monkeypatch):
    d = doc()
    serve(monkeypatch, {services.SERVICES_URL: d, SIG_URL: sign(d)})
    s = services.services()
    assert s["_source"] == "live" and s["llm"]["base_url"] == "https://gw.example.org"
    assert s["llm"]["models"] == ["test-model:cloud", "other:cloud"]
    cache = services._cache_path()
    assert cache.read_bytes() == d and cache.with_suffix(".sig").is_file()
    monkeypatch.setattr(services, "_download", lambda *a: pytest.fail("must not fetch"))
    s2 = services.services()
    assert s2["_source"] == "cache" and s2["llm"] == s["llm"]
    # the hosted helpers follow the document
    from morie import datahub, hosted

    hosted.reset_probe_cache()
    assert hosted.hosted_base_url() == "https://gw.example.org"
    assert hosted.hosted_auth_url() == "https://gw.example.org/auth"
    assert hosted.hosted_model() == "test-model:cloud"
    assert datahub.data_url() == "https://tables.example.org"
    # env overrides still win
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "https://mine.example.org/")
    monkeypatch.setenv("MORIE_HOSTED_MODEL", "local:latest")
    monkeypatch.setenv("MORIE_DATA_URL", "https://d.example.org")
    assert hosted.hosted_base_url() == "https://mine.example.org"
    assert hosted.hosted_model() == "local:latest"
    assert datahub.data_url() == "https://d.example.org"
    monkeypatch.setenv("MORIE_HOSTED_BASE_URL", "off")
    assert hosted.hosted_base_url() is None


def test_stale_cache_refreshes_and_an_older_live_document_is_a_rollback(svc, monkeypatch):
    old = doc(issued="2026-10-01T00:00:00Z", base="https://old.example.org")
    cache = services._cache_path()
    services._write(cache, old, sign(old))
    import os

    os.utime(cache, (time.time() - 3 * 86400, time.time() - 3 * 86400))
    newer = doc(issued="2026-10-05T00:00:00Z", base="https://new.example.org")
    serve(monkeypatch, {services.SERVICES_URL: newer, SIG_URL: sign(newer)})
    s = services.services()
    assert s["_source"] == "live" and s["llm"]["base_url"] == "https://new.example.org"
    older = doc(issued="2026-09-01T00:00:00Z", base="https://rollback.example.org")
    serve(monkeypatch, {services.SERVICES_URL: older, SIG_URL: sign(older)})
    s = services.services(refresh=True)
    assert s["_source"] == "cache" and s["llm"]["base_url"] == "https://new.example.org"
    assert services._read(cache)["llm"]["base_url"] == "https://new.example.org"


def test_the_bundled_copy_is_the_floor_for_the_cache(svc, monkeypatch, tmp_path):
    bundled = doc(issued="2026-10-06T12:00:00Z", base="https://bundled.example.org")
    bpath = tmp_path / "bundled" / "morie-services.json"
    services._write(bpath, bundled, sign(bundled))
    monkeypatch.setattr(services, "_bundled_path", lambda: bpath)
    cache = services._cache_path()
    # older than the bundled copy: refused and deleted
    old = doc(issued="2026-10-01T00:00:00Z", base="https://old.example.org")
    services._write(cache, old, sign(old))
    s = services.services(offline=True)
    assert s["_source"] == "bundled" and s["llm"]["base_url"] == "https://bundled.example.org"
    assert not cache.exists()
    # the same date: the bundled copy serves and the cache stays
    services.forget()
    services._write(cache, bundled, sign(bundled))
    assert services.services(offline=True)["_source"] == "bundled"
    assert cache.exists()
    # newer: the cache serves
    services.forget()
    newer = doc(issued="2026-10-07T00:00:00Z", base="https://new.example.org")
    services._write(cache, newer, sign(newer))
    assert services.services(offline=True)["llm"]["base_url"] == "https://new.example.org"
    # a live document older than the bundled one is a rollback even with no cache at all
    services.forget()
    cache.unlink()
    cache.with_suffix(".sig").unlink()
    serve(monkeypatch, {services.SERVICES_URL: old, SIG_URL: sign(old)})
    assert services.services(refresh=True)["_source"] == "bundled"
    assert not cache.exists()


def test_a_document_that_does_not_verify_is_ignored_at_every_step(svc, monkeypatch):
    d = doc()
    _, other_sk = mldsa_keygen(44, seed=bytes([1]) * 32)
    bad = {
        "other_key": sign(d, sk=other_sk),
        "other_context": sign(d, context=b"morie-services-v2"),
        "other_scheme": sign(d).replace("ML-DSA-44", "ML-DSA-65"),
        "tampered": sign(d.replace(b"gw.example.org", b"evil.example.org")),
        "not_json": "not a signature",
        "not_hex": json.dumps({"scheme": "ML-DSA-44", "context": "morie-services", "signature": "zz"}),
        "empty": "",
    }
    for name, sig in bad.items():
        assert not services.verify(d, sig), name
        serve(monkeypatch, {services.SERVICES_URL: d, SIG_URL: sig})
        services.forget()
        s = services.services(refresh=True)
        # the bundled copy is signed with the real key, not this test's: nothing is left but "off"
        assert s["_source"] == "off" and s["llm"]["mode"] == "off", name
        assert not services._cache_path().exists(), name
    from morie import datahub, hosted

    hosted.reset_probe_cache()
    assert hosted.hosted_base_url() is None
    with pytest.raises(RuntimeError, match="not available right now.*rmorie.com/access"):
        datahub.data_url()
    for pairs in ({}, {services.SERVICES_URL: d}, {services.SERVICES_URL: b"", SIG_URL: sign(d)}):
        serve(monkeypatch, pairs)
        assert services.fetch(5) is None
    monkeypatch.setenv("MORIE_SERVICES_URL", "http://127.0.0.1/morie-services.json")
    monkeypatch.setattr(services, "_download", lambda *a: pytest.fail("must not fetch"))
    assert services.fetch(5) is None


def test_parser_accepts_only_a_sane_v1_document():
    assert services.parse(doc()) is not None
    assert services.parse(doc(version=2)) is None
    assert services.parse(doc(llm_mode="anonymous")) is None
    assert services.parse(doc(data_mode="open")) is None
    assert services.parse(doc(issued="yesterday")) is None
    assert services.parse(doc(base="http://gw.example.org")) is None
    assert services.parse(doc(data_base="https://user:pw@tables.example.org")) is None
    assert services.parse(b"[]") is None
    assert services.parse(b'{"version":1}') is None
    assert services.parse(b"{not json") is None
    assert services.parse(b"\xff\xfe") is None
    off = services.parse(doc(llm_mode="off", data_mode="off", base="", data_base=""))
    assert off["llm"]["mode"] == "off" and off["llm"]["models"] == ["test-model:cloud", "other:cloud"]


def test_offline_memo_is_per_process_and_a_refresh_replaces_it(svc, monkeypatch):
    d = doc()
    serve(monkeypatch, {services.SERVICES_URL: d, SIG_URL: sign(d)})
    assert services.services(offline=True)["_source"] == "off"
    assert services.services(offline=True)["_source"] == "off"
    assert services.services()["_source"] == "live"
    assert services.services(offline=True)["_source"] == "live"
    assert "rmorie.com/access" in services.access_hint()


def test_the_signature_url_of_a_mirror_without_a_json_suffix():
    assert services._sig_url("https://rmorie.com/.well-known/morie-services.json") == (
        "https://rmorie.com/.well-known/morie-services.sig"
    )
    assert services._sig_url("https://mirror.example.org/services") == "https://mirror.example.org/services.sig"
