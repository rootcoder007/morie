# morie.fn -- internal core (rootcoder007/morie)
r"""BLAKE2b, natively.

RFC 7693. Written out here because morie's fn tree takes no external
imports, and because Argon2 (:mod:`secarg`) is defined on top of it --
a hash that is not bit-exact would make every Argon2 vector fail with
no indication of which of the two was wrong, so this one is anchored
separately.

Two details carry the parameters into the state and are easy to miss:
the digest length, key length and fanout are XORed into ``h[0]`` as a
**parameter block**, so BLAKE2b-256 is not a truncation of
BLAKE2b-512; and a **key** is padded to a full 128-byte block and
prepended to the message, so keyed hashing needs no separate HMAC
construction.

References
----------
Saarinen, M.-J. & Aumasson, J.-P. (2015) "The BLAKE2 Cryptographic
Hash and Message Authentication Code (MAC)", RFC 7693,
doi:10.17487/RFC7693. Sec. 2.1 (the IV, taken from SHA-512), Sec. 2.7
(the SIGMA message schedule and the 12 rounds of BLAKE2b), Sec. 3.1
(the mixing function G with the rotation constants 32, 24, 16, 63),
Sec. 3.2 (the compression function F, its column and diagonal mixes,
the finalisation flag and the XOR of both halves of v into h), Sec.
3.3 (the parameter block XORed into h[0] and the key padded to a full
block and prepended), and Appendix A, the worked BLAKE2b-512 example
used as the anchor.
"""

__all__ = ["blake2b", "BLOCK_SIZE", "MAX_DIGEST"]

_MASK64 = 0xffffffffffffffff
BLOCK_SIZE = 128
MAX_DIGEST = 64

_IV = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
]

_SIGMA = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
]


def _as_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(bytearray(int(v) & 0xff for v in data))


def _rotr(x, n):
    return ((x >> n) | (x << (64 - n))) & _MASK64


def _G(v, a, b, c, d, x, y):
    v[a] = (v[a] + v[b] + x) & _MASK64
    v[d] = _rotr(v[d] ^ v[a], 32)
    v[c] = (v[c] + v[d]) & _MASK64
    v[b] = _rotr(v[b] ^ v[c], 24)
    v[a] = (v[a] + v[b] + y) & _MASK64
    v[d] = _rotr(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & _MASK64
    v[b] = _rotr(v[b] ^ v[c], 63)


def _compress(h, block, t, last):
    m = [int.from_bytes(bytes(block[i:i + 8]), "little")
         for i in range(0, 128, 8)]
    v = list(h) + list(_IV)
    v[12] ^= t & _MASK64
    v[13] ^= (t >> 64) & _MASK64
    if last:
        v[14] ^= _MASK64
    for r in range(12):
        s = _SIGMA[r % 10]
        _G(v, 0, 4, 8, 12, m[s[0]], m[s[1]])
        _G(v, 1, 5, 9, 13, m[s[2]], m[s[3]])
        _G(v, 2, 6, 10, 14, m[s[4]], m[s[5]])
        _G(v, 3, 7, 11, 15, m[s[6]], m[s[7]])
        _G(v, 0, 5, 10, 15, m[s[8]], m[s[9]])
        _G(v, 1, 6, 11, 12, m[s[10]], m[s[11]])
        _G(v, 2, 7, 8, 13, m[s[12]], m[s[13]])
        _G(v, 3, 4, 9, 14, m[s[14]], m[s[15]])
    for i in range(8):
        h[i] ^= v[i] ^ v[i + 8]


def blake2b(data=b"", digest_size=64, key=b""):
    r"""BLAKE2b with an arbitrary digest length and optional key.

    The digest and key lengths enter through the PARAMETER BLOCK, so
    BLAKE2b-256 is a different function from a truncated BLAKE2b-512.
    """
    n = int(digest_size)
    k = _as_bytes(key)
    if not 1 <= n <= MAX_DIGEST:
        raise ValueError("_blake2: the digest size must lie in "
                         "1..64, got %d" % n)
    if len(k) > MAX_DIGEST:
        raise ValueError("_blake2: the key may be at most 64 bytes, "
                         "got %d" % len(k))
    h = list(_IV)
    h[0] ^= 0x01010000 ^ (len(k) << 8) ^ n
    msg = bytearray()
    if k:
        msg += bytearray(k) + bytearray(BLOCK_SIZE - len(k))
    msg += bytearray(_as_bytes(data))
    if not msg:
        msg = bytearray(BLOCK_SIZE)
        _compress(h, msg, 0, True)
    else:
        total = len(msg)
        pos = 0
        while total - pos > BLOCK_SIZE:
            pos += BLOCK_SIZE
            _compress(h, msg[pos - BLOCK_SIZE:pos], pos, False)
        tail = bytearray(msg[pos:])
        counted = total
        tail += bytearray(BLOCK_SIZE - len(tail))
        _compress(h, tail, counted, True)
    out = bytearray()
    for x in h:
        out += x.to_bytes(8, "little")
    return bytes(out[:n])
