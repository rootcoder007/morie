# morie.fn -- function file (rootcoder007/morie)
"""ML-DSA (FIPS 204) post-quantum signature -- verify."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_verify(message: bytes, signature: bytes, pk: bytes, context: bytes = b"") -> CryptoResult:
    """Verify an ML-DSA signature (FIPS 204); a signature made by rmorie or OpenSSL verifies too.

    :param message: Original message bytes.
    :param signature: Signature from mldsa_sign().
    :param pk: Public key from mldsa_keygen().
    :param context: The context string the signature was made with.
    :return: CryptoResult with valid flag.

    >>> from morie.fn.mldsa import mldsa_keygen
    >>> from morie.fn.mldss import mldsa_sign
    >>> keys = mldsa_keygen(44, seed=bytes(32))
    >>> sig = mldsa_sign(b"manifest", keys.extra["sk"]).extra["signature"]
    >>> mldsa_verify(b"manifest", sig, keys.extra["pk"]).extra["valid"]
    True
    """
    from morie.crypto._dilithium import mldsa_verify as _verify

    valid = _verify(message, signature, pk, context)
    return CryptoResult(
        algorithm="ML-DSA",
        operation="verify",
        success=valid,
        extra={"valid": valid},
    )


mldsv = mldsa_verify


def cheatsheet() -> str:
    return "mldsa_verify({}) -> ML-DSA (FIPS 204) post-quantum signature -- verify."


# compact alias per ledger/NAMING.md
mldsaverify = mldsa_verify
