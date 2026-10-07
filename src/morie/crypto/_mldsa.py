"""ML-DSA (FIPS 204), in pure Python.

The three parameter sets of FIPS 204 (August 2024): ML-DSA-44, ML-DSA-65 and ML-DSA-87, with
the byte encodings of the standard, so keys and signatures interoperate with any conforming
implementation (rmoriebricklayer's native one, OpenSSL 3.5, liboqs). Signing is "pure" ML-DSA
(Algorithm 2) with a context string of at most 255 bytes; the hedged variant draws 32 fresh
bytes per signature, the deterministic one uses 32 zero bytes (FIPS 204, section 3.4).

Checked byte for byte against OpenSSL 3.5.7's deterministic signatures (the vectors
rmoriebricklayer uses) and against rmoriebricklayer for key generation from a seed.
"""

from __future__ import annotations

import hashlib
import os

__all__ = [
    "PARAMS",
    "keygen",
    "keygen_internal",
    "sign",
    "verify",
    "level_of_public_key",
    "level_of_secret_key",
]

Q = 8380417
N = 256
D = 13
ZETA = 1753

# name: (k, ell, eta, tau, lambda, gamma1, gamma2, omega)
PARAMS = {
    44: (4, 4, 2, 39, 128, 1 << 17, (Q - 1) // 88, 80),
    65: (6, 5, 4, 49, 192, 1 << 19, (Q - 1) // 32, 55),
    87: (8, 7, 2, 60, 256, 1 << 19, (Q - 1) // 32, 75),
}


def _brv8(k: int) -> int:
    return int(f"{k:08b}"[::-1], 2)


_ZETAS = [pow(ZETA, _brv8(k), Q) for k in range(256)]
_N_INV = pow(256, -1, Q)  # 8347681


def _H(data: bytes, n: int) -> bytes:
    return hashlib.shake_256(data).digest(n)


class _Xof:
    """An incremental SHAKE reader (hashlib's digest(n) restarts, so read in growing blocks)."""

    def __init__(self, ctor, data: bytes):
        self._h = ctor(data)
        self._buf = b""
        self._pos = 0

    def read(self, n: int) -> bytes:
        while self._pos + n > len(self._buf):
            self._buf = self._h.digest(max(2 * len(self._buf), 1024))
        out = self._buf[self._pos : self._pos + n]
        self._pos += n
        return out


# ---- number theory --------------------------------------------------------------------------


def _mod_pm(r: int, alpha: int) -> int:
    """r mod+- alpha: the representative in (-alpha/2, alpha/2]."""
    r0 = r % alpha
    if r0 > alpha // 2:
        r0 -= alpha
    return r0


def _ntt(w: list[int]) -> list[int]:
    a = list(w)
    m = 0
    length = 128
    while length >= 1:
        start = 0
        while start < 256:
            m += 1
            z = _ZETAS[m]
            for j in range(start, start + length):
                t = z * a[j + length] % Q
                a[j + length] = (a[j] - t) % Q
                a[j] = (a[j] + t) % Q
            start += 2 * length
        length //= 2
    return a


def _intt(w: list[int]) -> list[int]:
    a = list(w)
    m = 256
    length = 1
    while length < 256:
        start = 0
        while start < 256:
            m -= 1
            z = -_ZETAS[m]
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % Q
                a[j + length] = z * (t - a[j + length]) % Q
            start += 2 * length
        length *= 2
    return [x * _N_INV % Q for x in a]


def _pmul(a: list[int], b: list[int]) -> list[int]:
    return [x * y % Q for x, y in zip(a, b)]


def _padd(a: list[int], b: list[int]) -> list[int]:
    return [(x + y) % Q for x, y in zip(a, b)]


def _psub(a: list[int], b: list[int]) -> list[int]:
    return [(x - y) % Q for x, y in zip(a, b)]


def _mat_vec_ntt(A, v_hat):
    out = []
    for row in A:
        acc = [0] * N
        for a, v in zip(row, v_hat):
            acc = _padd(acc, _pmul(a, v))
        out.append(acc)
    return out


def _power2round(r: int) -> tuple[int, int]:
    rp = r % Q
    r0 = _mod_pm(rp, 1 << D)
    return (rp - r0) >> D, r0


def _decompose(r: int, gamma2: int) -> tuple[int, int]:
    rp = r % Q
    r0 = _mod_pm(rp, 2 * gamma2)
    if rp - r0 == Q - 1:
        return 0, r0 - 1
    return (rp - r0) // (2 * gamma2), r0


def _high_bits(r: int, gamma2: int) -> int:
    return _decompose(r, gamma2)[0]


def _low_bits(r: int, gamma2: int) -> int:
    return _decompose(r, gamma2)[1]


def _make_hint(z: int, r: int, gamma2: int) -> int:
    return int(_high_bits(r, gamma2) != _high_bits(r + z, gamma2))


def _use_hint(h: int, r: int, gamma2: int) -> int:
    m = (Q - 1) // (2 * gamma2)
    r1, r0 = _decompose(r, gamma2)
    if h == 1 and r0 > 0:
        return (r1 + 1) % m
    if h == 1 and r0 <= 0:
        return (r1 - 1) % m
    return r1


def _inf_norm(polys) -> int:
    best = 0
    for p in polys:
        for c in p:
            c = c % Q
            v = Q - c if c > Q // 2 else c
            if v > best:
                best = v
    return best


# ---- bit packing (little-endian bit order, FIPS 204 Algorithms 9-21) -------------------------


def _pack_bits(values: list[int], width: int) -> bytes:
    acc = 0
    nbits = 0
    out = bytearray()
    for v in values:
        acc |= v << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_bits(data: bytes, width: int, count: int) -> list[int]:
    acc = int.from_bytes(data, "little")
    mask = (1 << width) - 1
    return [(acc >> (i * width)) & mask for i in range(count)]


def _simple_bit_pack(w: list[int], b: int) -> bytes:
    return _pack_bits(w, b.bit_length())


def _simple_bit_unpack(v: bytes, b: int) -> list[int]:
    return _unpack_bits(v, b.bit_length(), N)


def _bit_pack(w: list[int], a: int, b: int) -> bytes:
    """Coefficients in [-a, b] stored as b - w."""
    return _pack_bits([(b - (x if x <= Q // 2 else x - Q)) for x in (c % Q for c in w)], (a + b).bit_length())


def _bit_unpack(v: bytes, a: int, b: int) -> list[int]:
    return [b - x for x in _unpack_bits(v, (a + b).bit_length(), N)]


def _hint_bit_pack(h, omega: int, k: int) -> bytes:
    y = bytearray(omega + k)
    index = 0
    for i in range(k):
        for j in range(N):
            if h[i][j]:
                y[index] = j
                index += 1
        y[omega + i] = index
    return bytes(y)


def _hint_bit_unpack(y: bytes, omega: int, k: int):
    h = [[0] * N for _ in range(k)]
    index = 0
    for i in range(k):
        if y[omega + i] < index or y[omega + i] > omega:
            return None
        first = index
        while index < y[omega + i]:
            if index > first and y[index - 1] >= y[index]:
                return None
            h[i][y[index]] = 1
            index += 1
    for i in range(index, omega):
        if y[i] != 0:
            return None
    return h


# ---- sampling (FIPS 204 Algorithms 29-34) ----------------------------------------------------


def _sample_in_ball(rho: bytes, tau: int) -> list[int]:
    xof = _Xof(hashlib.shake_256, rho)
    s = xof.read(8)
    h = int.from_bytes(s, "little")
    c = [0] * N
    for i in range(N - tau, N):
        j = xof.read(1)[0]
        while j > i:
            j = xof.read(1)[0]
        c[i] = c[j]
        c[j] = Q - 1 if (h >> (i + tau - N)) & 1 else 1
    return c


def _rej_ntt_poly(rho: bytes) -> list[int]:
    xof = _Xof(hashlib.shake_128, rho)
    a = []
    while len(a) < N:
        b0, b1, b2 = xof.read(3)
        z = ((b2 & 0x7F) << 16) | (b1 << 8) | b0
        if z < Q:
            a.append(z)
    return a


def _coeff_from_half_byte(b: int, eta: int):
    if eta == 2 and b < 15:
        return 2 - (b % 5)
    if eta == 4 and b < 9:
        return 4 - b
    return None


def _rej_bounded_poly(rho: bytes, eta: int) -> list[int]:
    xof = _Xof(hashlib.shake_256, rho)
    a = []
    while len(a) < N:
        z = xof.read(1)[0]
        z0 = _coeff_from_half_byte(z & 0x0F, eta)
        z1 = _coeff_from_half_byte(z >> 4, eta)
        if z0 is not None:
            a.append(z0 % Q)
        if z1 is not None and len(a) < N:
            a.append(z1 % Q)
    return a


def _expand_a(rho: bytes, k: int, ell: int):
    return [[_rej_ntt_poly(rho + bytes([s, r])) for s in range(ell)] for r in range(k)]


def _expand_s(rho: bytes, k: int, ell: int, eta: int):
    s1 = [_rej_bounded_poly(rho + r.to_bytes(2, "little"), eta) for r in range(ell)]
    s2 = [_rej_bounded_poly(rho + (r + ell).to_bytes(2, "little"), eta) for r in range(k)]
    return s1, s2


def _expand_mask(rho: bytes, mu: int, ell: int, gamma1: int):
    c = 1 + (gamma1 - 1).bit_length()
    out = []
    for r in range(ell):
        v = _H(rho + (mu + r).to_bytes(2, "little"), 32 * c)
        out.append([x % Q for x in _bit_unpack(v, gamma1 - 1, gamma1)])
    return out


# ---- encodings -------------------------------------------------------------------------------


def _pk_encode(rho: bytes, t1) -> bytes:
    return rho + b"".join(_simple_bit_pack(p, (1 << ((Q - 1).bit_length() - D)) - 1) for p in t1)


def _pk_decode(pk: bytes, k: int):
    rho = pk[:32]
    w = 32 * ((Q - 1).bit_length() - D)
    t1 = [
        _simple_bit_unpack(pk[32 + i * w : 32 + (i + 1) * w], (1 << ((Q - 1).bit_length() - D)) - 1) for i in range(k)
    ]
    return rho, t1


def _sk_encode(rho, K, tr, s1, s2, t0, eta) -> bytes:
    out = rho + K + tr
    out += b"".join(_bit_pack(p, eta, eta) for p in s1)
    out += b"".join(_bit_pack(p, eta, eta) for p in s2)
    out += b"".join(_bit_pack(p, (1 << (D - 1)) - 1, 1 << (D - 1)) for p in t0)
    return out


def _sk_decode(sk: bytes, k: int, ell: int, eta: int):
    rho, K, tr = sk[:32], sk[32:64], sk[64:128]
    pos = 128
    we = 32 * (2 * eta).bit_length()
    s1 = []
    for _ in range(ell):
        s1.append([x % Q for x in _bit_unpack(sk[pos : pos + we], eta, eta)])
        pos += we
    s2 = []
    for _ in range(k):
        s2.append([x % Q for x in _bit_unpack(sk[pos : pos + we], eta, eta)])
        pos += we
    wt = 32 * D
    t0 = []
    for _ in range(k):
        t0.append([x % Q for x in _bit_unpack(sk[pos : pos + wt], (1 << (D - 1)) - 1, 1 << (D - 1))])
        pos += wt
    return rho, K, tr, s1, s2, t0


def _sig_encode(c_tilde: bytes, z, h, gamma1: int, omega: int, k: int) -> bytes:
    return c_tilde + b"".join(_bit_pack(p, gamma1 - 1, gamma1) for p in z) + _hint_bit_pack(h, omega, k)


def _sig_decode(sig: bytes, k: int, ell: int, lam: int, gamma1: int, omega: int):
    cl = lam // 4
    c_tilde = sig[:cl]
    wz = 32 * (1 + (gamma1 - 1).bit_length())
    z = []
    pos = cl
    for _ in range(ell):
        z.append([x % Q for x in _bit_unpack(sig[pos : pos + wz], gamma1 - 1, gamma1)])
        pos += wz
    h = _hint_bit_unpack(sig[pos:], omega, k)
    return c_tilde, z, h


def _w1_encode(w1, gamma2: int) -> bytes:
    return b"".join(_simple_bit_pack(p, (Q - 1) // (2 * gamma2) - 1) for p in w1)


def _sizes(level: int) -> tuple[int, int, int]:
    k, ell, eta, tau, lam, gamma1, gamma2, omega = PARAMS[level]
    pk = 32 + 32 * k * ((Q - 1).bit_length() - D)
    sk = 128 + 32 * ((k + ell) * (2 * eta).bit_length() + D * k)
    sig = lam // 4 + ell * 32 * (1 + (gamma1 - 1).bit_length()) + omega + k
    return pk, sk, sig


def level_of_public_key(pk: bytes) -> int:
    for lv in PARAMS:
        if len(pk) == _sizes(lv)[0]:
            return lv
    raise ValueError(f"not an ML-DSA public key: {len(pk)} bytes (1312, 1952 or 2592 expected)")


def level_of_secret_key(sk: bytes) -> int:
    for lv in PARAMS:
        if len(sk) == _sizes(lv)[1]:
            return lv
    raise ValueError(f"not an ML-DSA secret key: {len(sk)} bytes (2560, 4032 or 4896 expected)")


# ---- the three algorithms --------------------------------------------------------------------


def keygen_internal(xi: bytes, level: int = 65) -> tuple[bytes, bytes]:
    """ML-DSA.KeyGen_internal (Algorithm 6): the key pair a 32-byte seed determines."""
    if level not in PARAMS:
        raise ValueError("level must be 44, 65 or 87")
    if len(xi) != 32:
        raise ValueError("the seed must be 32 bytes")
    k, ell, eta, *_ = PARAMS[level]
    seed = _H(xi + bytes([k, ell]), 128)
    rho, rho_p, K = seed[:32], seed[32:96], seed[96:]
    A = _expand_a(rho, k, ell)
    s1, s2 = _expand_s(rho_p, k, ell, eta)
    s1_hat = [_ntt(p) for p in s1]
    t = [_padd(_intt(row), s) for row, s in zip(_mat_vec_ntt(A, s1_hat), s2)]
    t1, t0 = [], []
    for p in t:
        hi, lo = zip(*(_power2round(c) for c in p))
        t1.append(list(hi))
        t0.append([x % Q for x in lo])
    pk = _pk_encode(rho, t1)
    tr = _H(pk, 64)
    sk = _sk_encode(rho, K, tr, s1, s2, t0, eta)
    return pk, sk


def keygen(level: int = 65, seed: bytes | None = None) -> tuple[bytes, bytes]:
    """ML-DSA.KeyGen (Algorithm 1): a key pair from a fresh (or the given) 32-byte seed."""
    return keygen_internal(os.urandom(32) if seed is None else bytes(seed), level)


def _sign_internal(sk: bytes, m_prime: bytes, rnd: bytes) -> bytes:
    level = level_of_secret_key(sk)
    k, ell, eta, tau, lam, gamma1, gamma2, omega = PARAMS[level]
    rho, K, tr, s1, s2, t0 = _sk_decode(sk, k, ell, eta)
    s1_hat = [_ntt(p) for p in s1]
    s2_hat = [_ntt(p) for p in s2]
    t0_hat = [_ntt(p) for p in t0]
    A = _expand_a(rho, k, ell)
    mu = _H(tr + m_prime, 64)
    rho_pp = _H(K + rnd + mu, 64)
    kappa = 0
    beta = tau * eta
    while True:
        y = _expand_mask(rho_pp, kappa, ell, gamma1)
        w = [_intt(p) for p in _mat_vec_ntt(A, [_ntt(p) for p in y])]
        w1 = [[_high_bits(c, gamma2) for c in p] for p in w]
        c_tilde = _H(mu + _w1_encode(w1, gamma2), lam // 4)
        c_hat = _ntt(_sample_in_ball(c_tilde, tau))
        cs1 = [_intt(_pmul(c_hat, p)) for p in s1_hat]
        cs2 = [_intt(_pmul(c_hat, p)) for p in s2_hat]
        z = [_padd(a, b) for a, b in zip(y, cs1)]
        w_minus = [_psub(a, b) for a, b in zip(w, cs2)]
        r0 = [[_low_bits(c, gamma2) % Q for c in p] for p in w_minus]
        kappa += ell
        if _inf_norm(z) >= gamma1 - beta or _inf_norm(r0) >= gamma2 - beta:
            continue
        ct0 = [_intt(_pmul(c_hat, p)) for p in t0_hat]
        h = [[_make_hint((-a) % Q, (b + a) % Q, gamma2) for a, b in zip(pc, pw)] for pc, pw in zip(ct0, w_minus)]
        if _inf_norm(ct0) >= gamma2 or sum(map(sum, h)) > omega:
            continue
        return _sig_encode(c_tilde, z, h, gamma1, omega, k)


def _verify_internal(pk: bytes, m_prime: bytes, sig: bytes) -> bool:
    level = level_of_public_key(pk)
    k, ell, eta, tau, lam, gamma1, gamma2, omega = PARAMS[level]
    if len(sig) != _sizes(level)[2]:
        return False
    rho, t1 = _pk_decode(pk, k)
    c_tilde, z, h = _sig_decode(sig, k, ell, lam, gamma1, omega)
    if h is None:
        return False
    if _inf_norm(z) >= gamma1 - tau * eta:
        return False
    A = _expand_a(rho, k, ell)
    tr = _H(pk, 64)
    mu = _H(tr + m_prime, 64)
    c_hat = _ntt(_sample_in_ball(c_tilde, tau))
    az = _mat_vec_ntt(A, [_ntt(p) for p in z])
    ct1 = [_pmul(c_hat, _ntt([(c << D) % Q for c in p])) for p in t1]
    w_approx = [_intt(_psub(a, b)) for a, b in zip(az, ct1)]
    w1 = [[_use_hint(hh, c, gamma2) for hh, c in zip(ph, pw)] for ph, pw in zip(h, w_approx)]
    return _H(mu + _w1_encode(w1, gamma2), lam // 4) == c_tilde


def _m_prime(message: bytes, context: bytes) -> bytes:
    if len(context) > 255:
        raise ValueError("the context string is at most 255 bytes")
    return bytes([0, len(context)]) + context + message


def sign(sk: bytes, message: bytes, context: bytes = b"", deterministic: bool = False) -> bytes:
    """ML-DSA.Sign (Algorithm 2), pure: hedged by default, deterministic on request."""
    rnd = bytes(32) if deterministic else os.urandom(32)
    return _sign_internal(bytes(sk), _m_prime(bytes(message), bytes(context)), rnd)


def verify(pk: bytes, message: bytes, signature: bytes, context: bytes = b"") -> bool:
    """ML-DSA.Verify (Algorithm 3), pure."""
    if len(context) > 255:
        return False
    try:
        return _verify_internal(bytes(pk), _m_prime(bytes(message), bytes(context)), bytes(signature))
    except ValueError:
        return False
