"""Hybrid KEM-DEM encryption (ML-KEM-768 + ChaCha20-Poly1305).

Combines post-quantum key encapsulation (ML-KEM-768) with symmetric
authenticated encryption (ChaCha20-Poly1305) via HKDF-SHA256 key
derivation.

A random 32-byte symmetric key encrypts the payload; it is wrapped under a
key derived from ``HKDF(shared_secret || kem_ct || pk)``, where the shared
secret is the ML-KEM-768 (FIPS 203) encapsulation's, so only the holder of
the secret key can unwrap it. (morie / rmorie 1.3.x derived the wrapping key
from ``kem_ct || pk`` alone, which anyone holding the file and the public key
can compute; such files still open, with a warning, and should be encrypted
again.) rmorie 1.4.0 writes and reads the same container.

Container format (all big-endian lengths)::

    "MORIEHYB" 0x02 [9B] || len(kem_ct) [4B] || kem_ct ||
    wrapped_key_nonce [12B] || wrapped_key_ct [32B] || wrapped_key_tag [16B] ||
    payload_nonce [12B] || aead_ct || payload_tag [16B]

WARNING: Research/educational implementation. NOT constant-time.
For production use, prefer audited hybrid KEM libraries (e.g., liboqs).
"""

from __future__ import annotations

import hashlib
import os
import struct

from morie.crypto._chacha import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt
from morie.crypto._kdf import hkdf_sha256
from morie.crypto._mlkem import mlkem768_decaps, mlkem768_encaps, mlkem768_keygen

# Every container written by 1.4.0 starts with this marker. A 1.3.x container starts with the
# 4-byte length of the KEM ciphertext (0x00000440), so the two cannot be confused.
CONTAINER_MAGIC = b"MORIEHYB\x02"


class LegacyContainerWarning(UserWarning):
    """The file was written by morie 1.3.x, whose wrapping key was derivable from the ciphertext
    and the public key alone; it opened, and it should be encrypted again with this version."""


def container_version(ciphertext: bytes) -> int:
    """2 for a container written by morie 1.4.0 or later, 1 for the 1.3.x layout."""
    return 2 if ciphertext.startswith(CONTAINER_MAGIC) else 1


def _legacy_wrapping_key(kem_ct: bytes, pk: bytes) -> bytes:
    """The 1.3.x derivation, kept only to READ old files: no secret enters it."""
    return hkdf_sha256(
        kem_ct + pk,
        length=32,
        salt=hashlib.sha256(b"morie-hybrid-wrap-v1").digest(),
        info=b"key-wrap",
    )


def keygen() -> tuple[bytes, bytes]:
    """Generate an ML-KEM-768 key pair for hybrid encryption.

    Convenience wrapper around :func:`morie.crypto.mlkem768_keygen`.

    :return: ``(public_key, secret_key)`` as bytes.

    Examples
    --------
    >>> pk, sk = keygen()
    >>> (len(pk), len(sk))
    (1184, 2400)
    >>> hybrid_decrypt(hybrid_encrypt(b"the report", pk), sk)
    b'the report'
    """
    return mlkem768_keygen()


def _wrapping_key(shared_secret: bytes, kem_ct: bytes, pk: bytes) -> bytes:
    """Derive the 32-byte wrapping key from the KEM shared secret, bound to the ciphertext and key.

    The secret is what makes the wrap private: a key derived from ``kem_ct || pk`` alone is
    computable by anyone holding the ciphertext and the public key (the 1.3.x container did that,
    which is why files encrypted by 1.3.x cannot be opened by 1.4.0 and should be re-encrypted).
    """
    return hkdf_sha256(
        shared_secret + kem_ct + pk,
        length=32,
        salt=hashlib.sha256(b"morie-hybrid-wrap-v2").digest(),
        info=b"key-wrap",
    )


def hybrid_encrypt(plaintext: bytes, recipient_pk: bytes) -> bytes:
    """Encrypt data using hybrid ML-KEM-768 + ChaCha20-Poly1305.

    1. Encapsulate with the recipient's ML-KEM public key: a ciphertext and a shared secret.
    2. Derive the wrapping key from ``HKDF(shared_secret || kem_ct || pk)``.
    3. Generate a random 32-byte symmetric key, wrap it with ChaCha20-Poly1305.
    4. Encrypt the plaintext with the symmetric key, behind the 1.4.0 container marker.

    :param plaintext: Data to encrypt (arbitrary length).
    :param recipient_pk: Recipient's ML-KEM-768 public key.
    :return: Serialized ciphertext container.
    """
    kem_ct, shared_secret = mlkem768_encaps(recipient_pk)

    wrap_key = _wrapping_key(shared_secret, kem_ct, recipient_pk)
    sym_key = os.urandom(32)

    wrap_nonce = os.urandom(12)
    wrapped_ct, wrap_tag = chacha20_poly1305_encrypt(wrap_key, wrap_nonce, sym_key)

    payload_nonce = os.urandom(12)
    aead_ct, payload_tag = chacha20_poly1305_encrypt(sym_key, payload_nonce, plaintext)

    return (
        CONTAINER_MAGIC
        + struct.pack(">I", len(kem_ct))
        + kem_ct
        + wrap_nonce
        + wrapped_ct
        + wrap_tag
        + payload_nonce
        + aead_ct
        + payload_tag
    )


def hybrid_decrypt(ciphertext: bytes, recipient_sk: bytes) -> bytes:
    """Decrypt hybrid ML-KEM-768 + ChaCha20-Poly1305 ciphertext.

    :param ciphertext: Serialized container from :func:`hybrid_encrypt`.
    :param recipient_sk: Recipient's ML-KEM-768 secret key.
    :return: Decrypted plaintext.
    :raises ValueError: If the ciphertext is malformed or authentication fails.
    """
    legacy = container_version(ciphertext) == 1
    if not legacy:
        ciphertext = ciphertext[len(CONTAINER_MAGIC) :]
    if len(ciphertext) < 4:
        raise ValueError("Ciphertext too short to contain header")

    kem_ct_len = struct.unpack(">I", ciphertext[:4])[0]
    offset = 4

    min_len = offset + kem_ct_len + 12 + 32 + 16 + 12 + 16
    if len(ciphertext) < min_len:
        raise ValueError("Ciphertext too short")

    kem_ct = ciphertext[offset : offset + kem_ct_len]
    offset += kem_ct_len

    pk_start = 3 * 384
    pk_end = pk_start + 3 * 384 + 32
    recipient_pk = recipient_sk[pk_start:pk_end]

    if legacy:
        import warnings

        warnings.warn(
            "this file was encrypted by morie 1.3.x, whose wrapping key was derivable from the ciphertext "
            "and the public key alone; it opened, but anyone holding the file and the public key can open "
            "it too. Encrypt it again with this version (morie crypto encrypt FILE --to KEY).",
            LegacyContainerWarning,
            stacklevel=2,
        )
        wrap_key = _legacy_wrapping_key(kem_ct, recipient_pk)
    else:
        shared_secret = mlkem768_decaps(recipient_sk, kem_ct)
        wrap_key = _wrapping_key(shared_secret, kem_ct, recipient_pk)

    wrap_nonce = ciphertext[offset : offset + 12]
    offset += 12
    wrapped_ct = ciphertext[offset : offset + 32]
    offset += 32
    wrap_tag = ciphertext[offset : offset + 16]
    offset += 16

    sym_key = chacha20_poly1305_decrypt(wrap_key, wrap_nonce, wrapped_ct, wrap_tag)

    payload_nonce = ciphertext[offset : offset + 12]
    offset += 12
    payload_tag = ciphertext[-16:]
    aead_ct = ciphertext[offset:-16]

    return chacha20_poly1305_decrypt(sym_key, payload_nonce, aead_ct, payload_tag)
