# morie.fn -- function file (rootcoder007/morie)
"""ML-DSA (FIPS 204) post-quantum signature -- keygen."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_keygen(level: int = 65, seed: bytes | None = None) -> CryptoResult:
    """Generate an ML-DSA key pair (FIPS 204: ML-DSA-44, -65 or -87).

    :param level: 44, 65 (default) or 87.
    :param seed: Optional 32-byte seed; the same seed gives the same pair.
    :return: CryptoResult with pk and sk (bytes, the standard's encodings) in ``extra``.

    >>> r = mldsa_keygen(44, seed=bytes(32))
    >>> r.extra["pk_len"], r.extra["sk_len"]
    (1312, 2560)
    """
    from morie.crypto._dilithium import mldsa_keygen as _keygen

    pk, sk = _keygen(level, seed)
    return CryptoResult(
        algorithm="ML-DSA",
        operation="keygen",
        success=True,
        extra={"pk": pk, "sk": sk, "pk_len": len(pk), "sk_len": len(sk)},
    )


mldsa = mldsa_keygen


def cheatsheet() -> str:
    return "mldsa_keygen({}) -> ML-DSA (FIPS 204) post-quantum signature -- keygen."


# compact alias per ledger/NAMING.md
mldsakeygen = mldsa_keygen
