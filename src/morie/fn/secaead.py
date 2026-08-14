# morie.fn -- function file (rootcoder007/morie)
r"""AEAD_CHACHA20_POLY1305: encrypt and authenticate, in that order.

A stream cipher gives confidentiality and nothing else: flip a
ciphertext bit and exactly the corresponding plaintext bit flips,
undetected. AEAD closes that by pairing the cipher with a one-time
authenticator, and RFC 8439's construction is specific about how.

**ChaCha20** builds a 64-byte keystream block from a 16-word state --
four constants, eight key words, a block counter, three nonce words --
by twenty rounds of the quarter-round, alternating column and diagonal
rounds, then **adding the original state back**. That final addition
is what makes the block function non-invertible; without it the whole
thing would be a permutation an attacker could run backwards.

**Poly1305** is a one-time authenticator over
:math:`\mathbb{F}_{2^{130}-5}`, and its key must be **clamped**:
:math:`r[3], r[7], r[11], r[15]` keep only their low four bits and
:math:`r[4], r[8], r[12]` lose their low two. The clamp is not
decoration -- it bounds :math:`r` so the field arithmetic stays fast
and the security proof applies.

**The one-time key is derived per message.** ChaCha20 block **0** with
the same key and nonce produces it, and the message is then encrypted
starting at counter **1**. Reusing a Poly1305 key across two messages
lets an attacker solve for :math:`r` and forge at will, which is why
the derivation is part of the construction rather than left to the
caller.

**The MAC input is padded and length-tagged.** AAD, zero-padded to a
16-byte boundary; ciphertext, likewise; then the two lengths as
64-bit little-endian integers. Drop the length block and
:math:`(\text{AAD}, \text{ct})` boundaries become ambiguous -- an
attacker can move bytes from one field to the other and keep the tag
valid.

**Decryption verifies before it returns anything**, with a
constant-time comparison, and returns nothing at all on failure.

References
----------
Nir, Y. & Langley, A. (2018) "ChaCha20 and Poly1305 for IETF
Protocols", RFC 8439, doi:10.17487/RFC8439. Sec. 2.1-2.3 (the
quarter-round, the 16-word state layout with the constants
"expand 32-byte k", 20 rounds alternating column and diagonal, and
the addition of the original state to the permuted one); Sec. 2.5
(Poly1305 over 2^130 - 5 and the clamping of r -- r[3], r[7], r[11],
r[15] with their top four bits clear and r[4], r[8], r[12] with their
bottom two bits clear); Sec. 2.6 (the one-time Poly1305 key generated
from ChaCha20 block counter 0); Sec. 2.8 (the AEAD construction: AAD
padded to a 16-octet boundary, ciphertext padded likewise, then the
lengths of AAD and ciphertext as 64-bit little-endian numbers, with
encryption starting at counter 1); and Sec. 2.8.2, the test vector
this module is anchored on.

Bernstein, D. J. (2008) "ChaCha, a variant of Salsa20", *Workshop
Record of SASC 2008: The State of the Art of Stream Ciphers*. The
cipher.

Bernstein, D. J. (2005) "The Poly1305-AES message-authentication
code", *Fast Software Encryption (FSE 2005)*, LNCS 3557, 32-49,
doi:10.1007/11502760_3. The authenticator.
"""

from . import _sha2 as h
from ._richresult import RichResult

__all__ = ["chacha20_block", "chacha20", "poly1305_key_gen",
           "poly1305_mac", "aead_encrypt", "aead_decrypt"]

_MASK32 = 0xffffffff
_P1305 = (1 << 130) - 5
_CONST = (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)


def _rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & _MASK32


def _qr(s, a, b, c, d):
    s[a] = (s[a] + s[b]) & _MASK32
    s[d] = _rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _MASK32
    s[b] = _rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & _MASK32
    s[d] = _rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _MASK32
    s[b] = _rotl(s[b] ^ s[c], 7)


def _words_le(b):
    return [b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)
            | (b[i + 3] << 24) for i in range(0, len(b), 4)]


def _le_bytes(words):
    out = bytearray()
    for w in words:
        out += bytes([w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff,
                      (w >> 24) & 0xff])
    return bytes(out)


def chacha20_block(key, counter, nonce, rounds=20):
    r"""One 64-byte keystream block.

    The permuted state is ADDED to the original, which is what stops
    the block function from being invertible.
    """
    k = bytearray(h._as_bytes(key))
    n = bytearray(h._as_bytes(nonce))
    if len(k) != 32:
        raise ValueError("secaead: the key must be 32 bytes, got %d"
                         % len(k))
    if len(n) != 12:
        raise ValueError("secaead: the nonce must be 12 bytes, got "
                         "%d" % len(n))
    state = list(_CONST) + _words_le(k) + [int(counter) & _MASK32] \
        + _words_le(n)
    work = list(state)
    for _ in range(int(rounds) // 2):
        _qr(work, 0, 4, 8, 12)
        _qr(work, 1, 5, 9, 13)
        _qr(work, 2, 6, 10, 14)
        _qr(work, 3, 7, 11, 15)
        _qr(work, 0, 5, 10, 15)
        _qr(work, 1, 6, 11, 12)
        _qr(work, 2, 7, 8, 13)
        _qr(work, 3, 4, 9, 14)
    return _le_bytes([(work[i] + state[i]) & _MASK32
                      for i in range(16)])


def chacha20(key, counter, nonce, data):
    r"""XOR the data with the keystream from ``counter`` onward."""
    d = bytearray(h._as_bytes(data))
    out = bytearray()
    for i in range(0, len(d), 64):
        ks = bytearray(chacha20_block(key, int(counter) + i // 64,
                                      nonce))
        blk = d[i:i + 64]
        out += bytes(bytearray(blk[j] ^ ks[j]
                               for j in range(len(blk))))
    return bytes(out)


def _clamp(r):
    return r & 0x0ffffffc0ffffffc0ffffffc0fffffff


def poly1305_mac(message, key):
    r"""The one-time authenticator over :math:`2^{130}-5`.

    ``key`` is 32 bytes: the low 16 become :math:`r` (clamped) and the
    high 16 become :math:`s`.
    """
    k = bytearray(h._as_bytes(key))
    if len(k) != 32:
        raise ValueError("secaead: the Poly1305 key must be 32 "
                         "bytes, got %d" % len(k))
    r = _clamp(int.from_bytes(bytes(k[:16]), "little"))
    s = int.from_bytes(bytes(k[16:]), "little")
    m = bytearray(h._as_bytes(message))
    acc = 0
    for i in range(0, len(m), 16):
        blk = m[i:i + 16]
        n = int.from_bytes(bytes(blk) + b"\x01"
                           + b"\x00" * (16 - len(blk)), "little") \
            if len(blk) < 16 else int.from_bytes(
                bytes(blk) + b"\x01", "little")
        acc = ((acc + n) * r) % _P1305
    acc = (acc + s) & ((1 << 128) - 1)
    return acc.to_bytes(16, "little")


def poly1305_key_gen(key, nonce):
    r"""Block **0** gives the one-time key; the message starts at 1.

    Reusing a Poly1305 key across two messages lets an attacker solve
    for r and forge, so the derivation belongs to the construction.
    """
    return chacha20_block(key, 0, nonce)[:32]


def _pad16(b):
    return b"\x00" * ((16 - len(b) % 16) % 16)


def _mac_data(aad, ciphertext):
    a = h._as_bytes(aad)
    c = h._as_bytes(ciphertext)
    return (a + _pad16(a) + c + _pad16(c)
            + len(a).to_bytes(8, "little")
            + len(c).to_bytes(8, "little"))


def aead_encrypt(key, nonce, plaintext, aad=b""):
    r"""Encrypt from counter 1, then authenticate AAD and ciphertext."""
    otk = poly1305_key_gen(key, nonce)
    ct = chacha20(key, 1, nonce, plaintext)
    tag = poly1305_mac(_mac_data(aad, ct), otk)
    return RichResult(payload={
        "estimate": h.hexlify(ct), "ciphertext": ct,
        "ciphertext_hex": h.hexlify(ct), "tag": tag,
        "tag_hex": h.hexlify(tag), "onetime_key": otk,
        "aad_len": len(h._as_bytes(aad)), "ct_len": len(ct),
        "method": "AEAD_CHACHA20_POLY1305; Nir & Langley (2018) "
                  "RFC 8439",
        "note": "the length block is what keeps the AAD and "
                "ciphertext boundary unambiguous",
    })


def aead_decrypt(key, nonce, ciphertext, tag, aad=b""):
    r"""Verify FIRST, in constant time, and return nothing on failure.

    Returning the plaintext alongside a "not authenticated" flag would
    invite exactly the use that breaks it.
    """
    otk = poly1305_key_gen(key, nonce)
    want = poly1305_mac(_mac_data(aad, ciphertext), otk)
    if not h.constant_time_equal(want, tag):
        return {"valid": False, "plaintext": None,
                "note": "tag mismatch: nothing is returned, because a "
                        "caller given the plaintext anyway will use "
                        "it"}
    pt = chacha20(key, 1, nonce, ciphertext)
    return {"valid": True, "plaintext": pt,
            "expected_tag": want}


def cheatsheet():
    return ("secaead: a stream cipher alone lets an attacker flip a "
            "plaintext bit by flipping a ciphertext bit, undetected. "
            "ChaCha20 builds a 64-byte block from a 16-word state in "
            "20 rounds and ADDS the original state back -- that "
            "addition is what makes it non-invertible. Poly1305 is a "
            "ONE-TIME authenticator mod 2^130 - 5 whose key must be "
            "CLAMPED, and whose key is derived from ChaCha20 block 0 "
            "with the message encrypted from block 1, because reusing "
            "it across messages lets an attacker solve for r. The MAC "
            "input is AAD, pad16, ciphertext, pad16, then both lengths "
            "as 64-bit LE -- without the lengths the field boundary is "
            "ambiguous. Decryption verifies BEFORE returning anything.")


# compact alias per ledger/NAMING.md
chacha20poly1305 = aead_encrypt
