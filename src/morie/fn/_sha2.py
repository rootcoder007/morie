# morie.fn -- internal core (rootcoder007/morie)
r"""SHA-256 and HMAC-SHA-256, natively.

FIPS 180-4 for the hash and FIPS 198-1 / RFC 2104 for the MAC. Written
out here because morie's fn tree takes no external imports, and
because the published test vectors then become usable anchors: a hash
is either bit-exact against them or it is broken, with nothing in
between.

References
----------
National Institute of Standards and Technology (2015) *Secure Hash
Standard (SHS)*, FIPS PUB 180-4, doi:10.6028/NIST.FIPS.180-4. The
SHA-256 constants, message schedule and compression function.

National Institute of Standards and Technology (2008) *The Keyed-Hash
Message Authentication Code (HMAC)*, FIPS PUB 198-1,
doi:10.6028/NIST.FIPS.198-1; Krawczyk, H., Bellare, M. & Canetti, R.
(1997) "HMAC: Keyed-Hashing for Message Authentication", RFC 2104,
doi:10.17487/RFC2104. HMAC(K, m) = H((K' xor opad) || H((K' xor ipad)
|| m)), with K' the key padded to the block size or, if longer,
hashed first.
"""

__all__ = ["sha256", "hmac_sha256", "hexlify", "unhexlify",
           "constant_time_equal"]

_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
    0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
    0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
    0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
_H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
_MASK = 0xffffffff
BLOCK_SIZE = 64
DIGEST_SIZE = 32


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & _MASK


def _as_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(bytearray(int(v) & 0xff for v in data))


def sha256(data):
    r"""FIPS 180-4 SHA-256. Returns 32 raw bytes."""
    msg = bytearray(_as_bytes(data))
    ml = len(msg) * 8
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0x00)
    for i in range(7, -1, -1):
        msg.append((ml >> (8 * i)) & 0xff)
    h = list(_H0)
    for off in range(0, len(msg), 64):
        w = []
        for i in range(16):
            j = off + 4 * i
            w.append((msg[j] << 24) | (msg[j + 1] << 16)
                     | (msg[j + 2] << 8) | msg[j + 3])
        for i in range(16, 64):
            s0 = (_rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18)
                  ^ (w[i - 15] >> 3))
            s1 = (_rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19)
                  ^ (w[i - 2] >> 10))
            w.append((w[i - 16] + s0 + w[i - 7] + s1) & _MASK)
        a, b, c, d, e, f, g, hh = h
        for i in range(64):
            S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            ch = (e & f) ^ ((~e & _MASK) & g)
            t1 = (hh + S1 + ch + _K[i] + w[i]) & _MASK
            S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = (S0 + maj) & _MASK
            hh, g, f, e = g, f, e, (d + t1) & _MASK
            d, c, b, a = c, b, a, (t1 + t2) & _MASK
        h = [(x + y) & _MASK for x, y in
             zip(h, [a, b, c, d, e, f, g, hh])]
    out = bytearray()
    for x in h:
        out += bytes([(x >> 24) & 0xff, (x >> 16) & 0xff,
                      (x >> 8) & 0xff, x & 0xff])
    return bytes(out)


def hmac_sha256(key, message):
    r"""FIPS 198-1 / RFC 2104 HMAC with SHA-256.

    A key longer than the block size is HASHED first, not truncated --
    the step implementations most often get wrong.
    """
    k = _as_bytes(key)
    if len(k) > BLOCK_SIZE:
        k = sha256(k)
    k = k + b"\x00" * (BLOCK_SIZE - len(k))
    ipad = bytes(bytearray(b ^ 0x36 for b in bytearray(k)))
    opad = bytes(bytearray(b ^ 0x5c for b in bytearray(k)))
    return sha256(opad + sha256(ipad + _as_bytes(message)))


def hexlify(data):
    return "".join("%02x" % b for b in bytearray(_as_bytes(data)))


def unhexlify(text):
    s = str(text).replace(" ", "").replace("\n", "")
    if s[:2] in ("0x", "0X"):
        s = s[2:]
    if len(s) % 2:
        raise ValueError("_sha2: an odd number of hex digits")
    return bytes(bytearray(int(s[i:i + 2], 16)
                           for i in range(0, len(s), 2)))


def constant_time_equal(a, b):
    r"""Compare without leaking WHERE two values first differ.

    An early-exit comparison lets an attacker recover a tag byte by
    byte from timing; this one always reads every byte.
    """
    x = bytearray(_as_bytes(a))
    y = bytearray(_as_bytes(b))
    diff = len(x) ^ len(y)
    for i in range(max(len(x), len(y))):
        diff |= (x[i % len(x)] if x else 0) ^ (y[i % len(y)] if y
                                               else 0)
    return diff == 0
