# morie.fn -- function file (rootcoder007/morie)
r"""Argon2: make the attacker pay for memory, not just for time.

A password hash has to be slow, and iteration counts alone no longer
buy that: an attacker with GPUs or an ASIC parallelises pure
computation far more cheaply than the defender can. Argon2's answer is
to make the function **memory-hard** -- it fills :math:`m` kibibytes
and keeps referring back to them, so an attacker must either provision
that memory per guess or recompute blocks on demand, paying in time
for what they saved in space.

**Three parameters, and they are not interchangeable.** Memory
:math:`m`, passes :math:`t`, and lanes :math:`p`. The RFC is explicit
that memory is the primary defence and that lowering it in favour of
more passes weakens the time-space trade-off resistance --
``parameter_advice`` reports the RFC's own recommended configurations
rather than leaving the caller to guess.

**Three variants, for two different threats.**

* **Argon2d** picks its reference blocks from the data, which is
  fastest to resist time-space trade-offs -- and makes the access
  pattern depend on the password, so it leaks to a **side channel**.
* **Argon2i** picks them independently of the data, immune to that
  leak but weaker against trade-off attacks.
* **Argon2id** does the first half of the first pass the Argon2i way
  and everything after that the Argon2d way, which is the RFC's
  recommended default: the early, side-channel-visible portion is
  data-independent, and the rest gets the trade-off resistance.

The variant selection lives in one branch of ``_j_values`` and the
anchor checks it changes the tag, because a "variant" argument that
does nothing would be the easiest possible silent failure.

**Everything hangs off BLAKE2b.** The variable-length hash
:math:`H'`, the pre-hashing digest :math:`H_0` over every parameter
(so changing any of them changes the tag), and the compression
function :math:`G`, which applies the BLAKE2b round permutation to the
rows of a 1024-byte block and then to its columns -- rows alone would
never diffuse across the block.

References
----------
Biryukov, A., Dinu, D., Khovratovich, D. & Josefsson, S. (2021)
"Argon2 Memory-Hard Function for Password Hashing and Proof-of-Work
Applications", RFC 9106, doi:10.17487/RFC9106. [Spec fetched in
full.] Sec. 3.1 (the parameters and the pre-hashing digest H_0 over
p, T, m, t, v, y and the length-prefixed password, salt, secret and
associated data); Sec. 3.2 (block initialisation from H_0 and the
lane/column indices, and the XOR of the new block into the old one
from version 0x13 onward); Sec. 3.3 (the variable-length hash H');
Sec. 3.4.1.1-3.4.1.3 (Argon2d taking J_1, J_2 from the previous
block; Argon2i generating them from G(ZERO, G(ZERO, Z || LE64(i) ||
ZERO(968))) with Z carrying pass, lane, slice, m', t and type; and
Argon2id using the Argon2i rule when the pass is 0 and the slice is 0
or 1, and the Argon2d rule otherwise); Sec. 3.4.2 (the reference set
W, the lane l = J_2 mod p with the first slice of the first pass
taking the current lane, and the nonuniform mapping x = J_1^2 / 2^32,
y = |W| x / 2^32, zz = |W| - 1 - y); Sec. 3.5-3.6 (the compression
function G(X, Y) = R xor Q with P applied to rows then columns, and
the permutation P with its modified multiplication); Sec. 4 and 7.4
(parameter choice and the recommended configurations, with Argon2id as
the default); and Sec. 5, the test vectors this module is anchored on.

Biryukov, A., Dinu, D. & Khovratovich, D. (2016) "Argon2: New
Generation of Memory-Hard Functions for Password Hashing and Other
Applications", *2016 IEEE European Symposium on Security and Privacy
(EuroS&P)*, 292-302, doi:10.1109/EuroSP.2016.31. The design and the
trade-off analysis.

Saarinen, M.-J. & Aumasson, J.-P. (2015) "BLAKE2", RFC 7693,
doi:10.17487/RFC7693. The hash underneath; implemented in
:mod:`_blake2`.
"""

from . import _blake2 as b2
from . import _sha2 as h
from ._richresult import RichResult

__all__ = ["argon2", "variable_hash", "prehash", "compress",
           "parameter_advice"]

_MASK64 = 0xffffffffffffffff
_MASK32 = 0xffffffff
BLOCK = 1024
SL = 4
TYPES = {"argon2d": 0, "argon2i": 1, "argon2id": 2}
VERSION = 0x13


def _le32(n):
    return int(n).to_bytes(4, "little")


def _le64(n):
    return int(n).to_bytes(8, "little")


def variable_hash(data, length):
    r"""Argon2's :math:`H'`: BLAKE2b stretched past 64 bytes.

    Up to 64 bytes it is BLAKE2b of ``LE32(T) || A``. Beyond that it
    chains 64-byte hashes and keeps the FIRST 32 bytes of each, which
    is what stops the output from repeating.
    """
    T = int(length)
    if T < 1:
        raise ValueError("secarg: the output length must be "
                         "positive")
    a = h._as_bytes(data)
    if T <= 64:
        return b2.blake2b(_le32(T) + a, T)
    r = -(-T // 32) - 2
    out = bytearray()
    v = b2.blake2b(_le32(T) + a, 64)
    out += v[:32]
    for _ in range(1, r):
        v = b2.blake2b(v, 64)
        out += v[:32]
    v = b2.blake2b(v, T - 32 * r)
    out += v
    return bytes(out[:T])


def prehash(password, salt, parallelism, tag_length, memory, passes,
            variant="argon2id", secret=b"", associated=b"",
            version=VERSION):
    r""":math:`H_0`, over EVERY parameter.

    Changing the memory, the passes or the variant changes the digest,
    so a tag cannot silently be compared across configurations.
    """
    y = TYPES.get(str(variant))
    if y is None:
        raise ValueError("secarg: variant must be one of %s, got %r"
                         % (", ".join(sorted(TYPES)), variant))
    P = h._as_bytes(password)
    S = h._as_bytes(salt)
    K = h._as_bytes(secret)
    X = h._as_bytes(associated)
    if len(S) < 8:
        raise ValueError("secarg: the salt must be at least 8 bytes "
                         "(the RFC recommends 16), got %d" % len(S))
    buf = (_le32(parallelism) + _le32(tag_length) + _le32(memory)
           + _le32(passes) + _le32(version) + _le32(y)
           + _le32(len(P)) + P + _le32(len(S)) + S
           + _le32(len(K)) + K + _le32(len(X)) + X)
    return b2.blake2b(buf, 64)


def _gb(v, a, b, c, d):
    v[a] = (v[a] + v[b] + 2 * (v[a] & _MASK32) * (v[b] & _MASK32)) \
        & _MASK64
    v[d] = ((v[d] ^ v[a]) >> 32) | ((v[d] ^ v[a]) << 32) & _MASK64
    v[d] &= _MASK64
    v[c] = (v[c] + v[d] + 2 * (v[c] & _MASK32) * (v[d] & _MASK32)) \
        & _MASK64
    x = v[b] ^ v[c]
    v[b] = ((x >> 24) | (x << 40)) & _MASK64
    v[a] = (v[a] + v[b] + 2 * (v[a] & _MASK32) * (v[b] & _MASK32)) \
        & _MASK64
    x = v[d] ^ v[a]
    v[d] = ((x >> 16) | (x << 48)) & _MASK64
    v[c] = (v[c] + v[d] + 2 * (v[c] & _MASK32) * (v[d] & _MASK32)) \
        & _MASK64
    x = v[b] ^ v[c]
    v[b] = ((x >> 63) | (x << 1)) & _MASK64


def _P(v):
    _gb(v, 0, 4, 8, 12)
    _gb(v, 1, 5, 9, 13)
    _gb(v, 2, 6, 10, 14)
    _gb(v, 3, 7, 11, 15)
    _gb(v, 0, 5, 10, 15)
    _gb(v, 1, 6, 11, 12)
    _gb(v, 2, 7, 8, 13)
    _gb(v, 3, 4, 9, 14)


def compress(X, Y):
    r""":math:`G(X,Y)`: rows, then COLUMNS, then XOR back.

    Rows alone would leave each 128-byte strip independent; the column
    pass is what diffuses across the whole 1024-byte block.
    """
    R = [X[i] ^ Y[i] for i in range(128)]
    Q = list(R)
    for i in range(8):
        row = Q[16 * i:16 * i + 16]
        _P(row)
        Q[16 * i:16 * i + 16] = row
    for j in range(8):
        idx = []
        for i in range(8):
            idx += [16 * i + 2 * j, 16 * i + 2 * j + 1]
        col = [Q[k] for k in idx]
        _P(col)
        for t, k in enumerate(idx):
            Q[k] = col[t]
    return [Q[i] ^ R[i] for i in range(128)]


def _to_words(bs):
    return [int.from_bytes(bytes(bs[i:i + 8]), "little")
            for i in range(0, len(bs), 8)]


def _to_bytes(ws):
    out = bytearray()
    for w in ws:
        out += w.to_bytes(8, "little")
    return bytes(out)


def _addresses(pass_no, lane, slice_no, m_prime, passes, y, counter):
    zero = [0] * 128
    inp = [0] * 128
    inp[0] = pass_no
    inp[1] = lane
    inp[2] = slice_no
    inp[3] = m_prime
    inp[4] = passes
    inp[5] = y
    inp[6] = counter
    return compress(zero, compress(zero, inp))


def argon2(password, salt, memory=32, passes=3, parallelism=4,
           tag_length=32, variant="argon2id", secret=b"",
           associated=b""):
    r"""The full function. ``memory`` is in kibibytes.

    Returns the tag plus the parameters it was computed under, since a
    tag compared across different parameters is meaningless.
    """
    y = TYPES.get(str(variant))
    if y is None:
        raise ValueError("secarg: variant must be one of %s, got %r"
                         % (", ".join(sorted(TYPES)), variant))
    p = int(parallelism)
    t = int(passes)
    m = int(memory)
    if p < 1:
        raise ValueError("secarg: parallelism must be at least 1")
    if t < 1:
        raise ValueError("secarg: at least one pass is required")
    if m < 8 * p:
        raise ValueError("secarg: memory must be at least 8*p = %d "
                         "KiB, got %d" % (8 * p, m))
    m_prime = (m // (SL * p)) * (SL * p)
    q = m_prime // p
    seg = q // SL
    H0 = prehash(password, salt, p, tag_length, m, t, variant,
                 secret, associated)
    B = [[None] * q for _ in range(p)]
    for i in range(p):
        B[i][0] = _to_words(variable_hash(H0 + _le32(0) + _le32(i),
                                          BLOCK))
        B[i][1] = _to_words(variable_hash(H0 + _le32(1) + _le32(i),
                                          BLOCK))
    for r in range(t):
        for sl in range(SL):
            for i in range(p):
                data_indep = (y == 1) or (y == 2 and r == 0
                                          and sl < 2)
                addr, counter = None, 0
                start = 0
                if r == 0 and sl == 0:
                    start = 2
                    if data_indep:
                        counter += 1
                        addr = _addresses(r, i, sl, m_prime, t, y,
                                          counter)
                for idx in range(start, seg):
                    if data_indep and idx % 128 == 0:
                        counter += 1
                        addr = _addresses(r, i, sl, m_prime, t, y,
                                          counter)
                    j = sl * seg + idx
                    prev = B[i][j - 1] if j > 0 else B[i][q - 1]
                    if data_indep:
                        pr = addr[idx % 128]
                    else:
                        pr = prev[0]
                    J1 = pr & _MASK32
                    J2 = (pr >> 32) & _MASK32
                    lane = i if (r == 0 and sl == 0) else J2 % p
                    if r == 0:
                        if sl == 0 or lane == i:
                            W = j - 1
                        else:
                            W = sl * seg - (1 if idx == 0 else 0)
                    else:
                        if lane == i:
                            W = q - seg + idx - 1
                        else:
                            W = q - seg - (1 if idx == 0 else 0)
                    if W < 1:
                        W = 1
                    x = (J1 * J1) >> 32
                    yy = (W * x) >> 32
                    zz = W - 1 - yy
                    startpos = 0 if r == 0 else \
                        ((sl + 1) % SL) * seg
                    ref = (startpos + zz) % q
                    new = compress(prev, B[lane][ref])
                    if r == 0:
                        B[i][j] = new
                    else:
                        B[i][j] = [new[k] ^ B[i][j][k]
                                   for k in range(128)]
    C = list(B[0][q - 1])
    for i in range(1, p):
        C = [C[k] ^ B[i][q - 1][k] for k in range(128)]
    tag = variable_hash(_to_bytes(C), int(tag_length))
    return RichResult(payload={
        "estimate": h.hexlify(tag), "tag": tag,
        "tag_hex": h.hexlify(tag), "variant": variant,
        "memory_kib": m, "memory_used_kib": m_prime, "passes": t,
        "parallelism": p, "version": VERSION,
        "data_independent_first_half": y == 2,
        "method": "Argon2 v1.3; Biryukov, Dinu, Khovratovich & "
                  "Josefsson (2021) RFC 9106",
        "note": "a tag is only comparable against another computed "
                "under the SAME parameters, which is why they are "
                "returned with it",
    })


def parameter_advice(profile="first"):
    r"""The RFC's own recommended configurations.

    Memory is the primary defence; trading it for passes weakens
    exactly the property the function exists to provide.
    """
    rec = {
        "first": {"variant": "argon2id", "memory": 2 * 1024 * 1024,
                  "passes": 1, "parallelism": 4, "tag_length": 32,
                  "salt_bytes": 16,
                  "note": "RFC 9106 Sec. 4 first recommended option: "
                          "2 GiB, t = 1, p = 4"},
        "second": {"variant": "argon2id", "memory": 64 * 1024,
                   "passes": 3, "parallelism": 4, "tag_length": 32,
                   "salt_bytes": 16,
                   "note": "RFC 9106 Sec. 4 second option for memory-"
                           "constrained environments: 64 MiB, t = 3, "
                           "p = 4"},
    }
    if profile not in rec:
        raise ValueError("secarg: profile must be 'first' or "
                         "'second', got %r" % (profile,))
    out = dict(rec[profile])
    out["memory_gib"] = out["memory"] / (1024.0 * 1024.0)
    out["warning"] = ("lowering memory in favour of more passes "
                      "weakens time-space trade-off resistance")
    return out


def cheatsheet():
    return ("secarg: iteration counts no longer make a password hash "
            "slow -- a GPU or ASIC parallelises computation far more "
            "cheaply than the defender can. Argon2 is MEMORY-hard: "
            "fill m KiB and keep referring back, so an attacker "
            "provisions the memory per guess or recomputes blocks and "
            "pays in time. Memory is the PRIMARY parameter; trading it "
            "for passes weakens the trade-off resistance. THREE "
            "variants for TWO threats: 2d picks reference blocks from "
            "the DATA (best trade-off resistance, leaks through a side "
            "channel), 2i picks them independently (no leak, weaker), "
            "and 2id does the first half-pass the 2i way and the rest "
            "the 2d way -- the recommended default. G applies the "
            "BLAKE2b permutation to ROWS then COLUMNS; rows alone "
            "would not diffuse across the block.")


# compact alias per ledger/NAMING.md
argon2id = argon2

# public names resolved by fn/_lazy_map.json
argon2id_kdf = argon2
argon2idkdf = argon2
