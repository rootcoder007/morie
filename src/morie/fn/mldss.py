# morie.fn -- function file (rootcoder007/morie)
"""ML-DSA (FIPS 204) post-quantum signature -- sign."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_sign(message: bytes, sk: bytes, context: bytes = b"", deterministic: bool = False) -> CryptoResult:
    """Sign a message with ML-DSA (pure ML-DSA, FIPS 204).

    :param message: Message bytes (or str, as UTF-8).
    :param sk: Secret key from mldsa_keygen().
    :param context: Context string, at most 255 bytes.
    :param deterministic: The deterministic variant instead of the hedged default.
    :return: CryptoResult with signature in ``extra``.

    >>> from morie.fn.mldsa import mldsa_keygen
    >>> keys = mldsa_keygen(44, seed=bytes(32))
    >>> mldsa_sign(b"manifest", keys.extra["sk"]).extra["sig_len"]
    2420
    """
    from morie.crypto._dilithium import mldsa_sign as _sign

    sig = _sign(message, sk, context, deterministic)
    return CryptoResult(
        algorithm="ML-DSA",
        operation="sign",
        success=True,
        extra={"signature": sig, "sig_len": len(sig)},
    )


mldss = mldsa_sign


def cheatsheet() -> str:
    return "mldsa_sign({}) -> ML-DSA (FIPS 204) post-quantum signature -- sign."


# compact alias per ledger/NAMING.md
mldsasign = mldsa_sign
