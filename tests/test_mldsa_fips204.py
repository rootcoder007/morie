"""ML-DSA is FIPS 204: byte-identical to OpenSSL 3.5.7 and to rmoriebricklayer.

The OpenSSL vectors are the ones rmoriebricklayer tests against (deterministic signatures over
the message 01 02 ... 20 with the context "release"); the seeded key digests are
rmoriebricklayer's fips_keygen(<scheme>, seed = as.raw(0:31)).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from morie.crypto import mldsa_keygen, mldsa_sign, mldsa_verify

VECTORS = Path(__file__).parent / "data" / "mldsa_openssl_vectors.txt"
MSG = bytes(range(1, 33))
CTX = b"release"


def _vectors():
    for line in VECTORS.read_text().splitlines():
        if line.startswith("ML-DSA"):
            name, pk, sk, digest = line.split("|")
            yield name, bytes.fromhex(pk), bytes.fromhex(sk), digest


@pytest.mark.parametrize(
    "name,pk,sk,digest", list(_vectors()), ids=lambda v: v if isinstance(v, str) and v.startswith("ML") else ""
)
def test_deterministic_signature_matches_openssl(name, pk, sk, digest):
    sig = mldsa_sign(MSG, sk, context=CTX, deterministic=True)
    assert hashlib.sha256(sig).hexdigest() == digest
    assert mldsa_verify(MSG, sig, pk, context=CTX)
    assert not mldsa_verify(MSG + b"!", sig, pk, context=CTX)
    assert not mldsa_verify(MSG, sig, pk, context=b"other")
    assert not mldsa_verify(MSG, sig[:-1], pk, context=CTX)


@pytest.mark.parametrize(
    "level,pk_sha,sk_sha",
    [
        (
            44,
            "9f107644c1084526af3bc8098680b05499a2325a644e388fb4f970e058d19d46",
            "04bf6b9f579166a627961dfc5c3bf9717df868db88863856356c4668c8b56b0b",
        ),
        (
            65,
            "d666806e11cee19a7c989f7445f90dd419cf4d2d51db8c0fdb4c0f0a542238c9",
            "9f1e24f47795fe50040384e3d6183988047170fa2d866406b70fe0a3f8216063",
        ),
        (
            87,
            "91dc389cfaa01470b7f66eee45a4ae9026d154817c754dfe22298b3fa241ffcd",
            "764d3e223ed90c07bc91a0ab6ecd170e5c66ffe39f7039298596039a36005435",
        ),
    ],
)
def test_seeded_keygen_matches_rmoriebricklayer(level, pk_sha, sk_sha):
    pk, sk = mldsa_keygen(level, seed=bytes(range(32)))
    assert hashlib.sha256(pk).hexdigest() == pk_sha
    assert hashlib.sha256(sk).hexdigest() == sk_sha


def test_hedged_signatures_differ_and_verify():
    pk, sk = mldsa_keygen(65)
    s1, s2 = mldsa_sign(b"manifest", sk), mldsa_sign(b"manifest", sk)
    assert s1 != s2 and len(s1) == len(s2) == 3309
    assert mldsa_verify(b"manifest", s1, pk) and mldsa_verify("manifest", s2, pk)


def test_bad_inputs_are_refused_or_false():
    pk, sk = mldsa_keygen(44, seed=bytes(32))
    with pytest.raises(ValueError, match="44, 65 or 87"):
        mldsa_keygen(50)
    with pytest.raises(ValueError, match="32 bytes"):
        mldsa_keygen(44, seed=b"short")
    with pytest.raises(ValueError, match="secret key"):
        mldsa_sign(b"m", b"not a key")
    with pytest.raises(ValueError, match="255 bytes"):
        mldsa_sign(b"m", sk, context=bytes(256))
    sig = mldsa_sign(b"m", sk)
    assert mldsa_verify(b"m", sig, b"not a key") is False
    assert mldsa_verify(b"m", b"junk", pk) is False
    assert mldsa_verify(b"m", sig, pk, context=bytes(256)) is False
