# morie.fn -- function file (rootcoder007/morie)
"""Hamming code encode/decode."""

from __future__ import annotations

from . import _array_core as np

from ._containers import CryptoResult


def hamming_code(data: np.ndarray, r: int = 3, mode: str = "encode") -> CryptoResult:
    """Encode or decode a Hamming code.

    :param data: Binary vector (message for encode, received for decode).
    :param r: Number of parity bits.
    :param mode: "encode" or "decode".
    :return: CryptoResult with codeword/message in ``extra``.
    """
    from morie.crypto._ecc import hamming_decode, hamming_encode

    d = np.asarray(data, dtype=np.uint8).flatten()
    if mode == "encode":
        result = hamming_encode(d, r)
        return CryptoResult(
            algorithm="Hamming",
            operation="encode",
            success=True,
            extra=result,
        )
    result = hamming_decode(d, r)
    return CryptoResult(
        algorithm="Hamming",
        operation="decode",
        success=result["success"],
        extra=result,
    )


hamcd = hamming_code


def cheatsheet() -> str:
    return "hamming_code({}) -> Hamming code encode/decode."


# compact alias per ledger/NAMING.md
hammingcode = hamming_code
