"""ML-DSA signatures (FIPS 204): the morie.crypto entry points.

``mldsa_keygen``, ``mldsa_sign`` and ``mldsa_verify`` are the standard's ML-DSA-44/65/87 with its
byte encodings (``morie.crypto._mldsa``), so keys and signatures interoperate with rmorie and
rmoriebricklayer, OpenSSL 3.5 and liboqs. Until 1.4.0 these names ran a simplified
"Dilithium-lite" with JSON keys that no ML-DSA implementation could read or verify.
"""

from __future__ import annotations

from morie.crypto import _mldsa

__all__ = ["mldsa_keygen", "mldsa_sign", "mldsa_verify"]


def _as_bytes(x, what: str) -> bytes:
    if isinstance(x, str):
        return x.encode("utf-8")
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    raise TypeError(f"{what} must be bytes or str, not {type(x).__name__}")


def mldsa_keygen(level: int = 65, seed: bytes | None = None) -> tuple[bytes, bytes]:
    """Generate an ML-DSA key pair.

    :param level: 44, 65 (default) or 87 -- ML-DSA-44/65/87.
    :param seed: Optional 32-byte seed; the same seed gives the same key pair (FIPS 204 KeyGen_internal).
    :return: ``(public_key, secret_key)`` as bytes (1312/2560, 1952/4032 or 2592/4896 bytes).

    >>> pk, sk = mldsa_keygen(44, seed=bytes(32))
    >>> len(pk), len(sk)
    (1312, 2560)
    """
    return _mldsa.keygen(level, seed)


def mldsa_sign(message, sk_bytes: bytes, context: bytes = b"", deterministic: bool = False) -> bytes:
    """Sign a message with ML-DSA (pure ML-DSA, FIPS 204 Algorithm 2).

    :param message: Message (bytes, or str as UTF-8).
    :param sk_bytes: Secret key from :func:`mldsa_keygen`; its length gives the parameter set.
    :param context: Context string, at most 255 bytes (empty by default).
    :param deterministic: Use the deterministic variant (32 zero bytes of signing randomness)
        instead of the hedged default.
    :return: The signature (2420, 3309 or 4627 bytes).

    >>> pk, sk = mldsa_keygen(44, seed=bytes(32))
    >>> len(mldsa_sign(b"manifest", sk))
    2420
    """
    return _mldsa.sign(
        _as_bytes(sk_bytes, "sk_bytes"), _as_bytes(message, "message"), _as_bytes(context, "context"), deterministic
    )


def mldsa_verify(message, signature: bytes, pk_bytes: bytes, context: bytes = b"") -> bool:
    """Verify an ML-DSA signature; ``False`` for any signature, key or context that does not fit.

    >>> pk, sk = mldsa_keygen(44, seed=bytes(32))
    >>> sig = mldsa_sign(b"manifest", sk)
    >>> mldsa_verify(b"manifest", sig, pk), mldsa_verify(b"other", sig, pk)
    (True, False)
    """
    try:
        return _mldsa.verify(
            _as_bytes(pk_bytes, "pk_bytes"),
            _as_bytes(message, "message"),
            _as_bytes(signature, "signature"),
            _as_bytes(context, "context"),
        )
    except (TypeError, ValueError):
        return False
