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

__all__ = ["sha256", "hmac_sha256", "hexlify", "unhexlify", "constant_time_equal"]

_K = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]
_H0 = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
_MASK = 0xFFFFFFFF
BLOCK_SIZE = 64
DIGEST_SIZE = 32


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & _MASK


def _as_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(bytearray(int(v) & 0xFF for v in data))


def sha256(data):
    r"""FIPS 180-4 SHA-256. Returns 32 raw bytes."""
    msg = bytearray(_as_bytes(data))
    ml = len(msg) * 8
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0x00)
    for i in range(7, -1, -1):
        msg.append((ml >> (8 * i)) & 0xFF)
    h = list(_H0)
    for off in range(0, len(msg), 64):
        w = []
        for i in range(16):
            j = off + 4 * i
            w.append((msg[j] << 24) | (msg[j + 1] << 16) | (msg[j + 2] << 8) | msg[j + 3])
        for i in range(16, 64):
            s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
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
        h = [(x + y) & _MASK for x, y in zip(h, [a, b, c, d, e, f, g, hh])]
    out = bytearray()
    for x in h:
        out += bytes([(x >> 24) & 0xFF, (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF])
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
    opad = bytes(bytearray(b ^ 0x5C for b in bytearray(k)))
    return sha256(opad + sha256(ipad + _as_bytes(message)))


def hexlify(data):
    return "".join(f"{b:02x}" for b in bytearray(_as_bytes(data)))


def unhexlify(text):
    s = str(text).replace(" ", "").replace("\n", "")
    if s[:2] in ("0x", "0X"):
        s = s[2:]
    if len(s) % 2:
        raise ValueError("_sha2: an odd number of hex digits")
    return bytes(bytearray(int(s[i : i + 2], 16) for i in range(0, len(s), 2)))


def constant_time_equal(a, b):
    r"""Compare without leaking WHERE two values first differ.

    An early-exit comparison lets an attacker recover a tag byte by
    byte from timing; this one always reads every byte.
    """
    x = bytearray(_as_bytes(a))
    y = bytearray(_as_bytes(b))
    diff = len(x) ^ len(y)
    for i in range(max(len(x), len(y))):
        diff |= (x[i % len(x)] if x else 0) ^ (y[i % len(y)] if y else 0)
    return diff == 0
