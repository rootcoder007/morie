# SPDX-License-Identifier: AGPL-3.0-or-later
"""ML-KEM-768 decapsulation recovers the encapsulated secret, and the hybrid wrap needs the secret key."""

import pytest

from morie.crypto import _mlkem as M
from morie.crypto import hybrid


def _negacyclic(a, b):
    out = [0] * M.N
    for i in range(M.N):
        for j in range(M.N):
            k = i + j
            if k < M.N:
                out[k] = (out[k] + a[i] * b[j]) % M.Q
            else:
                out[k - M.N] = (out[k - M.N] - a[i] * b[j]) % M.Q
    return out


def test_ntt_is_a_ring_isomorphism():
    import random

    rng = random.Random(7)
    a = [rng.randrange(M.Q) for _ in range(M.N)]
    b = [rng.randrange(M.Q) for _ in range(M.N)]
    assert M._inv_ntt(M._ntt(a)) == a
    assert M._inv_ntt(M._poly_mul_ntt(M._ntt(a), M._ntt(b))) == _negacyclic(a, b)


def test_decaps_recovers_the_encapsulated_secret():
    for _ in range(3):
        pk, sk = M.mlkem768_keygen()
        assert (len(pk), len(sk)) == (1184, 2400)
        ct, ss = M.mlkem768_encaps(pk)
        assert len(ct) == 1088 and len(ss) == 32
        assert M.mlkem768_decaps(sk, ct) == ss


def test_tampered_ciphertext_gives_the_rejection_key_not_the_secret():
    pk, sk = M.mlkem768_keygen()
    ct, ss = M.mlkem768_encaps(pk)
    bad = bytes([ct[0] ^ 0x01]) + ct[1:]
    other = M.mlkem768_decaps(sk, bad)
    assert other != ss and len(other) == 32
    assert M.mlkem768_decaps(sk, bad) == other  # deterministic implicit rejection
    with pytest.raises(ValueError, match="ciphertext must be"):
        M.mlkem768_decaps(sk, ct[:-1])


def test_hybrid_needs_the_secret_key():
    pk, sk = hybrid.keygen()
    pk2, sk2 = hybrid.keygen()
    ct = hybrid.hybrid_encrypt(b"the report", pk)
    assert hybrid.hybrid_decrypt(ct, sk) == b"the report"
    # another party's secret key, and a forged secret key that only carries the public half, both fail
    with pytest.raises(ValueError):
        hybrid.hybrid_decrypt(ct, sk2)
    forged = b"\x00" * (3 * 384) + pk + M._sha3_256(pk) + b"\x00" * 32
    with pytest.raises(ValueError):
        hybrid.hybrid_decrypt(ct, forged)


def _legacy_container(plaintext: bytes, pk: bytes) -> bytes:
    """A container exactly as morie 1.3.x wrote it: the wrap key from the ciphertext and public key only."""
    import hashlib
    import os
    import struct

    from morie.crypto._chacha import chacha20_poly1305_encrypt
    from morie.crypto._kdf import hkdf_sha256

    kem_ct, _ = M.mlkem768_encaps(pk)
    wrap_key = hkdf_sha256(
        kem_ct + pk, length=32, salt=hashlib.sha256(b"morie-hybrid-wrap-v1").digest(), info=b"key-wrap"
    )
    sym_key, wrap_nonce, payload_nonce = os.urandom(32), os.urandom(12), os.urandom(12)
    wrapped, wrap_tag = chacha20_poly1305_encrypt(wrap_key, wrap_nonce, sym_key)
    aead, tag = chacha20_poly1305_encrypt(sym_key, payload_nonce, plaintext)
    return struct.pack(">I", len(kem_ct)) + kem_ct + wrap_nonce + wrapped + wrap_tag + payload_nonce + aead + tag


def test_files_from_1_3_still_open_and_say_so():
    pk, sk = hybrid.keygen()
    old = _legacy_container(b"a 1.3.x report", pk)
    assert hybrid.container_version(old) == 1
    with pytest.warns(hybrid.LegacyContainerWarning, match="encrypted by morie 1.3.x"):
        assert hybrid.hybrid_decrypt(old, sk) == b"a 1.3.x report"
    new = hybrid.hybrid_encrypt(b"a 1.4.0 report", pk)
    assert hybrid.container_version(new) == 2 and new.startswith(hybrid.CONTAINER_MAGIC)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a 1.4.0 file never triggers the legacy note
        assert hybrid.hybrid_decrypt(new, sk) == b"a 1.4.0 report"
