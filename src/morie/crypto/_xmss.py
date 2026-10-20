# SPDX-License-Identifier: AGPL-3.0-or-later
"""XMSS (RFC 8391) with SHA-256, n = 32, w = 16, in pure Python.

Single-tree XMSS as rmoriebricklayer's ``pqckg()`` / ``cpssign()`` build it: WOTS+ chain
seeds from ``PRF_keygen`` (NIST SP 800-208), the randomiser ``R = PRF(SK_PRF, toByte(idx, 32))``
and ``H_msg(R || root || toByte(idx, 32), M)``. At heights 10 and 16 these are the RFC's
``XMSS-SHA2_10_256`` and ``XMSS-SHA2_16_256``; the other heights (1 to 16) use the same
construction with a smaller or larger tree. Given the same seeds the root and every signature
are byte-identical to rmoriebricklayer's (and to the XMSS reference implementation's).

This module is stateless: the caller supplies the leaf index and must never sign two messages
at one index (:mod:`morie.capsule` keeps the used-index record that enforces it).

>>> s1, s2, s3 = bytes([1]) * 32, bytes([2]) * 32, bytes([3]) * 32
>>> root = keygen_root(s1, s2, 2)
>>> sig = sign(s1, s3, s2, 2, 0, b"message")
>>> sig["root"] == root.hex(), len(bytes.fromhex(sig["wire"]))
(True, 2244)
>>> verify(s2, root, 2, 0, b"message", sig["signature"], sig["auth"], sig["randomizer"])
True
>>> verify(s2, root, 2, 0, b"messagf", sig["signature"], sig["auth"], sig["randomizer"])
False
"""

from __future__ import annotations

import hashlib
import hmac
import struct

__all__ = ["keygen_root", "sign", "verify"]

N = 32
W = 16
LEN1 = 64
LEN2 = 3
LEN = LEN1 + LEN2

_D = [i.to_bytes(32, "big") for i in range(5)]  # toByte(dom, 32): F 0, H 1, H_msg 2, PRF 3, PRF_keygen 4
_sha = hashlib.sha256


def _adrs(t: int, w4: int = 0, w5: int = 0, w6: int = 0, w7: int = 0) -> bytes:
    # layer and tree address are zero in single-tree XMSS
    return struct.pack(">8I", 0, 0, 0, t, w4, w5, w6, w7)


def _prf(key: bytes, m32: bytes) -> bytes:
    return _sha(_D[3] + key + m32).digest()


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).to_bytes(len(a), "big")


def _rand_hash(left: bytes, right: bytes, seed: bytes, t: int, w4: int, w5: int, w6: int) -> bytes:
    key = _prf(seed, _adrs(t, w4, w5, w6, 0))
    bm0 = _prf(seed, _adrs(t, w4, w5, w6, 1))
    bm1 = _prf(seed, _adrs(t, w4, w5, w6, 2))
    return _sha(_D[1] + key + _xor(left, bm0) + _xor(right, bm1)).digest()


def _chain(x: bytes, start: int, steps: int, seed: bytes, ots: int, ci: int) -> bytes:
    for j in range(start, start + steps):
        key = _prf(seed, _adrs(0, ots, ci, j, 0))
        bm = _prf(seed, _adrs(0, ots, ci, j, 1))
        x = _sha(_D[0] + key + _xor(x, bm)).digest()
    return x


def _digits(msg32: bytes) -> list[int]:
    d = []
    for b in msg32:
        d += [b >> 4, b & 15]
    csum = sum(W - 1 - v for v in d) << 4
    cs = csum.to_bytes(2, "big")
    return d + [cs[0] >> 4, cs[0] & 15, cs[1] >> 4][:LEN2]


def _wots_sk(sk_seed: bytes, pub_seed: bytes, ots: int, i: int) -> bytes:
    return _sha(_D[4] + sk_seed + pub_seed + _adrs(0, ots, i, 0, 0)).digest()


def _ltree(pk: list[bytes], pub_seed: bytes, ots: int) -> bytes:
    height = 0
    while len(pk) > 1:
        nxt = [_rand_hash(pk[2 * i], pk[2 * i + 1], pub_seed, 1, ots, height, i) for i in range(len(pk) // 2)]
        if len(pk) % 2:
            nxt.append(pk[-1])
        pk = nxt
        height += 1
    return pk[0]


def _leaf(sk_seed: bytes, pub_seed: bytes, idx: int) -> bytes:
    pk = [_chain(_wots_sk(sk_seed, pub_seed, idx, i), 0, W - 1, pub_seed, idx, i) for i in range(LEN)]
    return _ltree(pk, pub_seed, idx)


def _tree(sk_seed: bytes, pub_seed: bytes, h: int, leaf: int) -> tuple[bytes, list[bytes]]:
    # ponytail: rebuilds the whole tree per signature (as rmoriebricklayer does); a BDS traversal
    # state would make signing O(h) if keys taller than ~12 ever matter
    lvl = [_leaf(sk_seed, pub_seed, i) for i in range(1 << h)]
    auth, pos = [], leaf
    for height in range(h):
        auth.append(lvl[pos ^ 1])
        lvl = [_rand_hash(lvl[2 * i], lvl[2 * i + 1], pub_seed, 2, 0, height, i) for i in range(len(lvl) // 2)]
        pos //= 2
    return lvl[0], auth


def _check(x: bytes, what: str) -> bytes:
    x = bytes(x)
    if len(x) != N:
        raise ValueError(f"`{what}` must be 32 bytes")
    return x


def _height(h: int) -> int:
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= 16:
        raise ValueError("`height` must be a single integer between 1 and 16")
    return h


def keygen_root(sk_seed: bytes, pub_seed: bytes, height: int) -> bytes:
    """The Merkle root (the public key, with ``pub_seed``) of the key these seeds determine."""
    return _tree(_check(sk_seed, "sk_seed"), _check(pub_seed, "pub_seed"), _height(height), 0)[0]


def sign(sk_seed: bytes, sk_prf: bytes, pub_seed: bytes, height: int, index: int, message: bytes) -> dict:
    """An XMSS signature at leaf ``index``, as hex fields: ``signature`` (the WOTS+ chains),
    ``auth``, ``index``, ``root``, ``randomizer`` and ``wire`` (the RFC 8391 byte string
    ``idx (4) || R || WOTS+ signature || auth path``)."""
    sk_seed, sk_prf, pub_seed = _check(sk_seed, "sk_seed"), _check(sk_prf, "sk_prf"), _check(pub_seed, "pub_seed")
    h = _height(height)
    if not isinstance(index, int) or not 0 <= index < (1 << h):
        raise ValueError(f"`index` exhausted: a height-{h} key signs 2^{h} messages")
    root, auth = _tree(sk_seed, pub_seed, h, index)
    r = _prf(sk_prf, index.to_bytes(32, "big"))
    dig = _sha(_D[2] + r + root + index.to_bytes(32, "big") + bytes(message)).digest()
    wots = b"".join(
        _chain(_wots_sk(sk_seed, pub_seed, index, i), 0, d, pub_seed, index, i) for i, d in enumerate(_digits(dig))
    )
    a = b"".join(auth)
    return {
        "signature": wots.hex(),
        "auth": a.hex(),
        "index": index,
        "root": root.hex(),
        "randomizer": r.hex(),
        "wire": (index.to_bytes(4, "big") + r + wots + a).hex(),
    }


def verify(
    pub_seed: bytes, root: bytes, height: int, index: int, message: bytes, signature: str, auth: str, randomizer: str
) -> bool:
    """``True`` when the hex ``signature``, ``auth`` and ``randomizer`` are a valid XMSS signature of
    ``message`` at ``index`` under ``root``; any malformed input is ``False``."""
    try:
        pub_seed, root = _check(pub_seed, "pub_seed"), _check(root, "root")
        h = _height(height)
        sig, ap, r = bytes.fromhex(signature), bytes.fromhex(auth), bytes.fromhex(randomizer)
    except (ValueError, TypeError):
        return False
    if len(sig) != LEN * N or len(ap) != h * N or len(r) != N or not isinstance(index, int):
        return False
    if not 0 <= index < (1 << h):
        return False
    dig = _sha(_D[2] + r + root + index.to_bytes(32, "big") + bytes(message)).digest()
    pk = [_chain(sig[i * N : (i + 1) * N], d, W - 1 - d, pub_seed, index, i) for i, d in enumerate(_digits(dig))]
    node, pos = _ltree(pk, pub_seed, index), index
    for k in range(h):
        sib = ap[k * N : (k + 1) * N]
        node = _rand_hash(*((node, sib) if pos % 2 == 0 else (sib, node)), pub_seed, 2, 0, k, pos // 2)
        pos //= 2
    return hmac.compare_digest(node, root)
