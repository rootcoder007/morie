# morie.fn -- function file (rootcoder007/morie)
r"""HKDF: extract entropy, then expand it -- two steps, on purpose.

Key derivation is usually written as one function, and RFC 5869's
argument is that it is really two, with different jobs.

**Extract** takes a source of keying material that is *not*
uniformly distributed -- a Diffie-Hellman shared secret, a
password-derived value, a hardware entropy pool -- and concentrates
whatever entropy it has into a fixed-length pseudorandom key:

.. math:: \mathrm{PRK} = \mathrm{HMAC}(\text{salt}, \mathrm{IKM}).

Note which argument is which. The **salt is the HMAC key** and the
input keying material is the *message*, which is backwards from
intuition and is exactly the step implementations get wrong. The salt
is optional, non-secret, and when absent is a string of zeros the
length of the hash.

**Expand** takes that PRK and produces as many bytes as needed:

.. math:: T(i) = \mathrm{HMAC}(\mathrm{PRK}, T(i-1)\,\|\,
          \text{info}\,\|\,i),

with :math:`T(0)` empty and the counter a single byte -- which caps
the output at :math:`255 \times \mathrm{HashLen}` bytes, a limit
this module enforces rather than silently wrapping.

**Why the split matters.** The ``info`` argument binds the derived key
to a context, so the same PRK yields *independent* keys for
independent purposes -- and the anchor demonstrates that a one-bit
change in ``info`` gives an unrelated key. Skip Extract when the input
is already a uniform key, and skipping is the documented
``expand``-only case, not a shortcut.

References
----------
Krawczyk, H. & Eronen, P. (2010) "HMAC-based Extract-and-Expand Key
Derivation Function (HKDF)", RFC 5869, doi:10.17487/RFC5869. Sec. 2.2
(Extract: PRK = HMAC-Hash(salt, IKM), with the salt as the HMAC key
and the IKM as the message, the salt optional and defaulting to
HashLen zeros); Sec. 2.3 (Expand: T(i) = HMAC-Hash(PRK, T(i-1) | info
| i) with T(0) the empty string and a single-octet counter, so
L <= 255*HashLen); Sec. 3.1 (the role of the salt); Sec. 3.2 (the
info field binding the output to application-specific context);
Sec. 3.3 (skipping Extract when the input is already a uniformly
random key); and Appendix A, the test vectors this module is anchored
on.

Krawczyk, H. (2010) "Cryptographic Extraction and Key Derivation: The
HKDF Scheme", *Advances in Cryptology - CRYPTO 2010*, LNCS 6223,
631-648, doi:10.1007/978-3-642-14623-7_34. The extract-then-expand
analysis.

National Institute of Standards and Technology (2008) *The Keyed-Hash
Message Authentication Code (HMAC)*, FIPS PUB 198-1,
doi:10.6028/NIST.FIPS.198-1. The MAC underneath; implemented in
:mod:`_sha2`.
"""

from . import _sha2 as h
from ._richresult import RichResult

__all__ = ["extract", "expand", "hkdf", "derive_context_keys"]

HASH_LEN = 32
MAX_BLOCKS = 255


def extract(ikm, salt=None):
    r""":math:`\mathrm{PRK} = \mathrm{HMAC}(\text{salt},
    \mathrm{IKM})`.

    The SALT is the HMAC key and the input keying material is the
    MESSAGE -- the way round that is easy to get backwards.
    """
    s = b"\x00" * HASH_LEN if salt is None else h._as_bytes(salt)
    return {"prk": h.hmac_sha256(s, ikm),
            "salt_supplied": salt is not None,
            "note": "the salt is the HMAC KEY; the IKM is the "
                    "message"}


def expand(prk, info=b"", length=32):
    r"""Counter-mode expansion, capped at :math:`255\times
    \mathrm{HashLen}`."""
    L = int(length)
    if L < 1:
        raise ValueError("seckdf: the output length must be "
                         "positive")
    if L > MAX_BLOCKS * HASH_LEN:
        raise ValueError("seckdf: L = %d exceeds 255*HashLen = %d; "
                         "the counter is a single octet, so this "
                         "cannot be satisfied"
                         % (L, MAX_BLOCKS * HASH_LEN))
    p = h._as_bytes(prk)
    if len(p) < HASH_LEN:
        raise ValueError("seckdf: the PRK is %d bytes, shorter than "
                         "the hash length %d -- Extract was probably "
                         "skipped on non-uniform input"
                         % (len(p), HASH_LEN))
    inf = h._as_bytes(info)
    out, t = bytearray(), b""
    i = 1
    while len(out) < L:
        t = h.hmac_sha256(p, t + inf + bytes([i]))
        out += t
        i += 1
    return {"okm": bytes(out[:L]), "blocks": i - 1, "length": L}


def hkdf(ikm, salt=None, info=b"", length=32, skip_extract=False):
    r"""Extract then expand -- or expand alone on an already-uniform
    key.

    ``skip_extract=True`` is the documented case for input that is
    already a uniformly random key, not a shortcut for saving a hash.
    """
    if skip_extract:
        prk = h._as_bytes(ikm)
        salted = False
    else:
        e = extract(ikm, salt)
        prk = e["prk"]
        salted = e["salt_supplied"]
    r = expand(prk, info, length)
    return RichResult(payload={
        "estimate": h.hexlify(r["okm"]), "okm": r["okm"],
        "okm_hex": h.hexlify(r["okm"]), "prk": prk,
        "prk_hex": h.hexlify(prk), "length": r["length"],
        "blocks": r["blocks"], "salt_supplied": salted,
        "extract_skipped": bool(skip_extract),
        "method": "HKDF-SHA256; Krawczyk & Eronen (2010) RFC 5869",
        "note": "info binds the output to a context, so one PRK gives "
                "independent keys for independent purposes",
    })


def derive_context_keys(ikm, contexts, salt=None, length=32):
    r"""One PRK, one key per context, all independent.

    The point of the ``info`` argument: deriving an encryption key and
    a MAC key from the same secret is safe precisely because their
    contexts differ.
    """
    e = extract(ikm, salt)
    keys = {}
    for c in contexts:
        keys[c] = expand(e["prk"], h._as_bytes(c), length)["okm"]
    hexed = {c: h.hexlify(v) for c, v in keys.items()}
    distinct = len(set(hexed.values())) == len(hexed)
    return {"keys": keys, "hex": hexed, "prk": e["prk"],
            "all_distinct": distinct,
            "note": "same PRK, different info, unrelated outputs"}


def cheatsheet():
    return ("seckdf: key derivation is TWO steps. EXTRACT concentrates "
            "a non-uniform secret into a fixed-length PRK: "
            "PRK = HMAC(salt, IKM) -- the SALT is the HMAC KEY and the "
            "IKM is the MESSAGE, which is the way round people get "
            "wrong. The salt is optional, non-secret, and defaults to "
            "HashLen zeros. EXPAND runs a counter mode: T(i) = "
            "HMAC(PRK, T(i-1) | info | i), capped at 255*HashLen "
            "because the counter is ONE octet. INFO binds the key to a "
            "context, so one PRK safely yields independent keys. Skip "
            "Extract only when the input is already uniform.")


# compact alias per ledger/NAMING.md
hkdf_sha256 = hkdf

# public names resolved by fn/_lazy_map.json
hkdf_extract_expand = hkdf
