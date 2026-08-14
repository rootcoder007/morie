"""Tests for the crypto batch: _sha2, seckdf (HKDF), sechsh.

Anchored on published test vectors -- FIPS 180-4, RFC 4231, RFC 5869
Appendix A and RFC 6962 -- so a passing test means bit-exactness.
"""
import importlib

import pytest


def M(name):
    return importlib.import_module("morie.fn." + name)


h = M("_sha2")
H = h.hexlify
U = h.unhexlify


# ----------------------------------------------------------------- _sha2
@pytest.mark.parametrize("msg,digest", [
    (b"abc",
     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    (b"",
     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
])
def test_sha256_published_vectors(msg, digest):
    assert H(h.sha256(msg)) == digest


@pytest.mark.parametrize("key,msg,mac", [
    (b"\x0b" * 20, b"Hi There",
     "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"),
    (b"Jefe", b"what do ya want for nothing?",
     "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"),
    (b"\xaa" * 131,
     b"Test Using Larger Than Block-Size Key - Hash Key First",
     "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"),
])
def test_hmac_rfc4231_vectors(key, msg, mac):
    assert H(h.hmac_sha256(key, msg)) == mac


def test_constant_time_equal_agrees_with_equality():
    assert h.constant_time_equal(b"abc", b"abc")
    assert not h.constant_time_equal(b"abc", b"abd")
    assert not h.constant_time_equal(b"abc", b"ab")


def test_hex_round_trip_and_odd_input():
    assert U(H(b"\x00\xff\x10")) == b"\x00\xff\x10"
    with pytest.raises(ValueError):
        U("abc")


# ---------------------------------------------------------------- seckdf
def test_seckdf_rfc5869_test_case_1():
    kd = M("seckdf")
    r = kd.hkdf(U("0b" * 22), U("000102030405060708090a0b0c"),
                U("f0f1f2f3f4f5f6f7f8f9"), 42)
    assert r["prk_hex"] == (
        "077709362c2e32df0ddc3f0dc47bba63"
        "90b6c73bb50f9c3122ec844ad7c2b3e5")
    assert r["okm_hex"] == (
        "3cb25f25faacd57a90434f64d0362f2a"
        "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865")


def test_seckdf_rfc5869_test_case_3_no_salt_no_info():
    kd = M("seckdf")
    r = kd.hkdf(U("0b" * 22), None, b"", 42)
    assert r["prk_hex"] == (
        "19ef24a32c717b167f33a91d6f648bdf"
        "96596776afdb6377ac434c1c293ccb04")
    assert r["okm_hex"] == (
        "8da4e775a563c18f715f802a063c5a31"
        "b8a11f5c5ee1879ec3454e5f3c738d2d"
        "9d201395faa4b61a96c8")


def test_seckdf_salt_is_the_hmac_key():
    kd = M("seckdf")
    assert kd.extract(b"ikm", b"salt")["prk"] == h.hmac_sha256(
        b"salt", b"ikm")
    assert kd.extract(b"ikm", b"salt")["prk"] != h.hmac_sha256(
        b"ikm", b"salt")


def test_seckdf_info_gives_independent_keys():
    kd = M("seckdf")
    r = kd.derive_context_keys(U("0b" * 22), ["enc", "mac", "enD"])
    assert r["all_distinct"]


def test_seckdf_enforces_the_counter_ceiling():
    kd = M("seckdf")
    assert len(kd.expand(U("00" * 32), b"", 255 * 32)["okm"]) == 8160
    with pytest.raises(ValueError):
        kd.expand(U("00" * 32), b"", 255 * 32 + 1)
    with pytest.raises(ValueError):
        kd.expand(U("00" * 32), b"", 0)


def test_seckdf_rejects_a_short_prk():
    kd = M("seckdf")
    with pytest.raises(ValueError):
        kd.expand(b"short", b"", 32)


# ---------------------------------------------------------------- sechsh
ENTRIES = [b"login alice", b"read record 7", b"delete record 7",
           b"logout alice"]


def test_sechsh_chain_is_the_stated_recurrence():
    hs = M("sechsh")
    c = hs.build_chain(ENTRIES)
    assert c["hashes"][0] == h.sha256(hs.GENESIS + ENTRIES[0])
    assert c["hashes"][1] == h.sha256(c["hashes"][0] + ENTRIES[1])


def test_sechsh_localises_tampering():
    hs = M("sechsh")
    c = hs.build_chain(ENTRIES)
    assert hs.verify_chain(ENTRIES, c["hashes"])["intact"]
    bad = list(ENTRIES)
    bad[2] = b"delete record 8"
    r = hs.verify_chain(bad, c["hashes"])
    assert not r["intact"]
    assert r["first_bad"] == 2
    assert r["verified_through"] == 2


def test_sechsh_is_only_tamper_evident():
    hs = M("sechsh")
    bad = list(ENTRIES)
    bad[2] = b"delete record 8"
    assert hs.verify_chain(bad,
                           hs.build_chain(bad)["hashes"])["intact"]


def test_sechsh_keyed_chain_differs():
    hs = M("sechsh")
    assert hs.build_chain(ENTRIES, key=b"secret")["head"] \
        != hs.build_chain(ENTRIES)["head"]


def test_sechsh_merkle_domain_separation():
    hs = M("sechsh")
    assert hs.merkle_root([]) == h.sha256(b"")
    assert hs.merkle_root([b"d"]) == h.sha256(b"\x00d")
    assert hs.merkle_root([b"a", b"b"]) == h.sha256(
        b"\x01" + h.sha256(b"\x00a") + h.sha256(b"\x00b"))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 13, 64])
def test_sechsh_every_leaf_proves_inclusion(n):
    hs = M("sechsh")
    log = [bytes([i]) for i in range(n)]
    root = hs.merkle_root(log)
    for i in range(n):
        pr = hs.inclusion_proof(log, i)
        assert hs.verify_inclusion(log[i], i, n, pr["path"],
                                   root)["valid"]


def test_sechsh_proof_is_logarithmic_and_fails_on_the_wrong_leaf():
    hs = M("sechsh")
    log = [bytes([i]) for i in range(64)]
    assert hs.inclusion_proof(log, 5)["length"] == 6
    root = hs.merkle_root(log)
    pr = hs.inclusion_proof(log, 0)
    assert not hs.verify_inclusion(b"\x63", 0, 64, pr["path"],
                                   root)["valid"]


def test_sechsh_rejects_bad_indices_and_lengths():
    hs = M("sechsh")
    log = [bytes([i]) for i in range(4)]
    with pytest.raises(ValueError):
        hs.inclusion_proof(log, 9)
    with pytest.raises(ValueError):
        hs.verify_chain(ENTRIES, hs.build_chain(ENTRIES)["hashes"][:3])
