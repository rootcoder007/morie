"""Tests for secrtt.rotating_token_envelope (envelope KEK rotation)."""

import os

import pytest

from morie.fn.secrtt import open_record, rotating_token_envelope, seal_record, unwrap_dek, wrap_dek


def test_secrtt_basic():
    """Rotation re-wraps every DEK under the new KEK (with the stored AAD)
    and touches no record: the old ciphertext still opens with the DEK
    recovered through the new KEK, and every unwrap is audited."""
    old, new = os.urandom(32), os.urandom(32)
    deks = [os.urandom(32) for _ in range(3)]
    wrapped = [wrap_dek(d, old, os.urandom(12), aad=b"table-a") for d in deks]
    sealed = seal_record(b"row payload", deks[1], os.urandom(12))
    log = []
    r = rotating_token_envelope(wrapped, old, new, [os.urandom(12) for _ in deks], audit_log=log)
    assert r["records_reencrypted"] == 0 and r["n"] == 3 and r["kek_id"] == "kek-2"
    assert [unwrap_dek(w, new)["dek"] for w in r["wrapped"]] == deks
    assert open_record(sealed, unwrap_dek(r["wrapped"][1], new)["dek"]) == b"row payload"
    assert [e["ok"] for e in log] == [True] * 3
    with pytest.raises(ValueError):
        unwrap_dek(r["wrapped"][0], old)


def test_secrtt_edge():
    """One nonce per DEK is required, and a DEK wrapped under another KEK
    fails authentication."""
    old = os.urandom(32)
    w = [wrap_dek(os.urandom(32), old, os.urandom(12))]
    with pytest.raises(ValueError):
        rotating_token_envelope(w, old, os.urandom(32), [])
    with pytest.raises(ValueError):
        rotating_token_envelope(w, os.urandom(32), os.urandom(32), [os.urandom(12)])
