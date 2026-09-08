"""Tests for _blake2 (RFC 7693) and secarg (Argon2, RFC 9106)."""
import importlib

import pytest

h = importlib.import_module("morie.fn._sha2")
b2 = importlib.import_module("morie.fn._blake2")
ar = importlib.import_module("morie.fn.secarg")
H = h.hexlify

P32 = bytes(bytearray([1] * 32))
S16 = bytes(bytearray([2] * 16))
K8 = bytes(bytearray([3] * 8))
X12 = bytes(bytearray([4] * 12))


def test_blake2b_rfc7693_abc():
    assert H(b2.blake2b(b"abc")) == (
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbff"
        "a2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386ed"
        "d4009923")


def test_blake2b_empty():
    assert H(b2.blake2b(b"")) == (
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f"
        "5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701a"
        "fe9be2ce")


def test_blake2b_length_is_not_truncation():
    assert H(b2.blake2b(b"abc", 32)) != H(b2.blake2b(b"abc"))[:64]


def test_blake2b_key_changes_the_digest():
    assert b2.blake2b(b"abc", 64, b"k") != b2.blake2b(b"abc", 64)


def test_blake2b_block_boundary():
    assert len(b2.blake2b(b"x" * 128)) == 64
    assert b2.blake2b(b"x" * 128) != b2.blake2b(b"x" * 127)


def test_blake2b_rejects_bad_sizes():
    with pytest.raises(ValueError):
        b2.blake2b(b"abc", 65)
    with pytest.raises(ValueError):
        b2.blake2b(b"abc", 0)
    with pytest.raises(ValueError):
        b2.blake2b(b"abc", 64, b"k" * 65)


def test_argon2_prehash_matches_rfc9106():
    got = H(ar.prehash(P32, S16, 4, 32, 32, 3, "argon2d", K8, X12))
    assert got.startswith("b8819791a0359660bb7709c85fa48f04")


@pytest.mark.parametrize("variant,tag", [
    ("argon2d", "512b391b6f1162975371d30919734294"
                "f868e3be3984f3c1a13a4db9fabe4acb"),
    ("argon2i", "c814d9d1dc7f37aa13f0d77f2494bda1"
                "c8de6b016dd388d29952a4c4672b6ce8"),
    ("argon2id", "0d640df58d78766c08c037a34a8b53c9"
                 "d01ef0452d75b65eb52520e96b01e659"),
])
def test_argon2_rfc9106_test_vectors(variant, tag):
    r = ar.argon2(P32, S16, memory=32, passes=3, parallelism=4,
                  tag_length=32, variant=variant, secret=K8,
                  associated=X12)
    assert r["tag_hex"] == tag


def test_argon2_parameters_are_bound_into_the_tag():
    a = ar.argon2(P32, S16, memory=32, passes=3, parallelism=4,
                  variant="argon2id", secret=K8, associated=X12)
    b = ar.argon2(P32, S16, memory=32, passes=4, parallelism=4,
                  variant="argon2id", secret=K8, associated=X12)
    c = ar.argon2(P32, S16, memory=32, passes=3, parallelism=4,
                  variant="argon2id", associated=X12)
    assert len({a["tag_hex"], b["tag_hex"], c["tag_hex"]}) == 3
    assert a["memory_kib"] == 32 and a["version"] == 0x13


def test_argon2_variable_hash_stretches_and_matches_below_64():
    assert ar.variable_hash(b"abc", 32) == b2.blake2b(
        (32).to_bytes(4, "little") + b"abc", 32)
    long_out = ar.variable_hash(b"abc", 128)
    assert len(long_out) == 128
    assert long_out[:32] != long_out[32:64]
    with pytest.raises(ValueError):
        ar.variable_hash(b"abc", 0)


def test_argon2_compress_diffuses_and_depends_on_the_xor():
    spike = [0] * 128
    spike[0] = 0x0123456789abcdef
    g = ar.compress(spike, [0] * 128)
    flat = ar.compress([0] * 128, [0] * 128)
    assert g[127] != flat[127]
    assert len(set(g)) > 100
    assert ar.compress([0] * 128, spike) == g
    assert ar.compress([w ^ 7 for w in spike], [7] * 128) == g


def test_argon2_rejects_bad_parameters():
    with pytest.raises(ValueError):
        ar.argon2(P32, b"short", memory=32)
    with pytest.raises(ValueError):
        ar.argon2(P32, S16, memory=4, parallelism=4)
    with pytest.raises(ValueError):
        ar.argon2(P32, S16, variant="argon2x")
    with pytest.raises(ValueError):
        ar.argon2(P32, S16, memory=32, passes=0)


def test_argon2_parameter_advice_matches_the_rfc():
    first = ar.parameter_advice("first")
    second = ar.parameter_advice("second")
    assert first["memory_gib"] == 2.0 and first["passes"] == 1
    assert first["variant"] == "argon2id"
    assert second["memory"] == 64 * 1024 and second["passes"] == 3
    with pytest.raises(ValueError):
        ar.parameter_advice("third")
