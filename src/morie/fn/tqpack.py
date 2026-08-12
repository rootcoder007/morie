r"""Bit-packing of quantiser indices.

Packs an array of ``b``-bit codebook indices into a dense byte string,
and unpacks it again. This is the storage half of a quantiser: choosing
4-bit codewords saves nothing if each is then stored in a 64-bit float.

The layout is **big-endian bit order within a big-endian byte stream**:
index 0 occupies the most significant ``b`` bits of byte 0, the next
index continues immediately after it, crossing byte boundaries without
padding. Only the final byte is zero-padded on the right. Fixing the
convention explicitly matters -- a reader that assumes the opposite bit
order recovers plausible-looking indices that are silently wrong, and
no checksum in the format would catch it.

Round-tripping is exact by construction, for every width and every
length, and that is the property the anchors check exhaustively rather
than on a sample.
"""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["pack_indices", "unpack_indices", "tqpack"]


def pack_indices(indices, bits):
    r"""Pack ``bits``-wide unsigned indices into bytes.

    Parameters
    ----------
    indices : array-like of int
        Values in ``[0, 2**bits - 1]``.
    bits : int
        Width per index, 1 to 32.

    Returns
    -------
    RichResult
        ``bytes`` is a list of integers in 0..255; ``n_bytes`` is
        ``ceil(n * bits / 8)``, the exact storage cost.
    """
    b = int(bits)
    if not (1 <= b <= 32):
        raise ValueError("pack_indices: bits must lie in 1..32, got %r" % (bits,))
    vals = []
    limit = (1 << b) - 1
    for v in np.atleast_1d(np.asarray(indices, dtype=float)):
        iv = int(v)
        if iv != v:
            raise ValueError("pack_indices: index %r is not an integer" % (v,))
        if iv < 0 or iv > limit:
            raise ValueError(
                "pack_indices: index %d does not fit in %d bits (max %d)"
                % (iv, b, limit))
        vals.append(iv)

    out = []
    acc = 0          # bit buffer, most significant bit first
    nbits = 0
    for iv in vals:
        acc = (acc << b) | iv
        nbits += b
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
            acc &= (1 << nbits) - 1
    if nbits:
        # Pad the tail on the RIGHT, so the last index keeps its
        # position in the stream.
        out.append((acc << (8 - nbits)) & 0xFF)

    n = len(vals)
    return RichResult(payload={
        "estimate": out,
        "bytes": out,
        "n_bytes": len(out),
        "n_indices": n,
        "bits": b,
        "bits_used": n * b,
        "padding_bits": len(out) * 8 - n * b,
        "compression_vs_float64": (64.0 * n / (8.0 * len(out))) if out else 0.0,
        "method": "Big-endian bit packing of fixed-width indices",
    })


def unpack_indices(data, bits, count):
    r"""Inverse of :func:`pack_indices`."""
    b = int(bits)
    if not (1 <= b <= 32):
        raise ValueError("unpack_indices: bits must lie in 1..32, got %r"
                         % (bits,))
    n = int(count)
    if n < 0:
        raise ValueError("unpack_indices: count must be non-negative")
    by = [int(v) & 0xFF for v in np.atleast_1d(np.asarray(data, dtype=float))]
    need = (n * b + 7) // 8
    if len(by) < need:
        raise ValueError(
            "unpack_indices: %d bytes cannot hold %d indices of %d bits "
            "(need %d)" % (len(by), n, b, need))

    out = []
    acc = 0
    nbits = 0
    pos = 0
    while len(out) < n:
        while nbits < b:
            acc = (acc << 8) | by[pos]
            pos += 1
            nbits += 8
        nbits -= b
        out.append((acc >> nbits) & ((1 << b) - 1))
        acc &= (1 << nbits) - 1
    return RichResult(payload={
        "estimate": out,
        "indices": out,
        "n_indices": len(out),
        "bits": b,
        "method": "Big-endian bit unpacking of fixed-width indices",
    })


def cheatsheet():
    return ("tqpack: pack b-bit indices big-endian, index 0 in the top "
            "b bits of byte 0, crossing byte boundaries; tail padded on "
            "the right; n_bytes = ceil(n*b/8); round-trip is exact.")


tqpack = pack_indices
