# morie.fn -- function file (rootcoder007/morie)
"""Coding-theory results from MacKay's *Information Theory, Inference, and Learning Algorithms*:
bounded-distance versus Shannon noise limits, self-dual codes, run-length-limited channel capacity,
the repetition-code error approximation, the robust soliton distribution of LT codes, the
decomposition of expected code length, and McGill's interaction information."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "bounded_distance_noise",
    "self_dual_code_check",
    "runlength_channel_capacity",
    "repetition_error_approx",
    "robust_soliton",
    "code_length_decomposition",
    "mcgill_interaction_information",
]


def _h2(f):
    return -f * math.log2(f) - (1 - f) * math.log2(1 - f)


def bounded_distance_noise(rate: float) -> RichResult:
    r"""Largest BSC noise level a rate-``R`` code can tolerate: Shannon's ``H2(f) = 1 - R`` versus ``f_bd = f / 2``.

    Shannon's decoder copes with any ``f`` below the root of ``H2(f) = 1 - R``
    (capacity ``C = 1 - H2(f)``, MacKay eq. 13.18-13.19); a bounded-distance
    decoder only with half of it (eq. 13.20). The root is found by bisection
    on ``(0, 1/2)``.

    References
    ----------
    MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning
    Algorithms*. Cambridge University Press, Section 13.8.

    Examples
    --------
    >>> r = bounded_distance_noise(0.5)
    >>> round(r.f_shannon, 12), round(r.f_bd, 12)
    (0.110027864438, 0.055013932219)
    """
    if not 0 < rate < 1:
        raise ValueError("rate must lie in (0, 1)")
    lo, hi = 1e-300, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _h2(mid) < 1 - rate:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    return RichResult(payload={"f_shannon": f, "f_bd": f / 2})


def self_dual_code_check(P) -> RichResult:
    r"""Is the systematic code with generator ``G = [I_K | P^T]`` self-dual? True iff ``P^T P = I_K`` (mod 2).

    A self-dual code has ``G G^T = 0`` (mod 2); for ``G = [I_K | P^T]`` that
    is ``I + P^T P = 0``, i.e. ``P`` is orthogonal modulo 2 (MacKay eq.
    13.42-13.43). Returns the verdict, ``P^T P mod 2`` and ``G G^T mod 2``.

    Examples
    --------
    >>> self_dual_code_check([[1, 1], [1, 0]]).self_dual
    False
    >>> self_dual_code_check([[0, 1], [1, 0]]).self_dual
    True
    """
    P = [[int(v) % 2 for v in row] for row in P]
    K = len(P)
    if any(len(row) != K for row in P):
        raise ValueError("P must be K x K")
    ptp = [[sum(P[r][i] * P[r][j] for r in range(K)) % 2 for j in range(K)] for i in range(K)]
    G = [[1 if i == j else 0 for j in range(K)] + [P[j][i] for j in range(K)] for i in range(K)]
    ggt = [[sum(G[i][c] * G[j][c] for c in range(2 * K)) % 2 for j in range(K)] for i in range(K)]
    ok = all(ptp[i][j] == (1 if i == j else 0) for i in range(K) for j in range(K))
    return RichResult(payload={"self_dual": ok, "PtP": ptp, "GGt": ggt})


def runlength_channel_capacity(L: int) -> RichResult:
    r"""Capacity (bits per symbol) of the binary channel forbidding runs of more than ``L`` ones.

    The capacity is the ``beta`` solving ``Z(beta) = sum_{l=1}^{L+1} 2^{-beta l} = 1``
    (MacKay eq. 17.29, reusing the variable-length channel of exercise 6.18);
    the simple code that appends a 0 after each maximal run has rate
    ``1 / (1 + 2^{-L})``, a lower bound.

    Examples
    --------
    >>> r = runlength_channel_capacity(1)
    >>> round(r.capacity, 12), round(r.simple_rate, 12)
    (0.694241913631, 0.666666666667)
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ssum(2.0 ** (-mid * ell) for ell in range(1, L + 2)) > 1:
            lo = mid
        else:
            hi = mid
    return RichResult(payload={"capacity": 0.5 * (lo + hi), "simple_rate": 1 / (1 + 2.0**-L)})


def repetition_error_approx(N: int, f: float, target: float | None = None) -> RichResult:
    r"""Leading-order error probability ``p_b = p_B ~ (4 f (1 - f))^{N/2}`` of the repetition code ``R_N`` (odd ``N``).

    From ``C(N, K) ~ 2^{N H2(K/N)}`` (MacKay eq. 1.38-1.39). With ``target``
    the blocklength ``N ~ 2 log(target) / log(4 f (1 - f))`` needed to reach it is returned too.

    Examples
    --------
    >>> r = repetition_error_approx(3, 0.1, target=1e-15)
    >>> round(r.pb, 12), round(r.n_target, 6)
    (0.216, 67.613633)
    """
    if not 0 < f < 0.5:
        raise ValueError("f must lie in (0, 1/2)")
    q = 4 * f * (1 - f)
    out = {"pb": q ** (N / 2)}
    if target is not None:
        out["n_target"] = 2 * math.log(target) / math.log(q)
    return RichResult(payload=out)


def robust_soliton(K: int, c: float = 0.1, delta: float = 0.5) -> RichResult:
    r"""Robust soliton degree distribution of Luby's LT fountain code.

    ``rho`` is the ideal soliton (``rho(1) = 1/K``, ``rho(d) = 1/(d(d-1))``);
    with ``S = c ln(K/delta) sqrt(K)`` and ``m = round(K/S)``,
    ``tau(d) = S / (K d)`` for ``d < m``, ``tau(m) = S ln(S/delta) / K``;
    ``mu = (rho + tau) / Z`` (MacKay eq. 50.2-50.4). For ``K = 10000``,
    ``c = 0.2``, ``delta = 0.05``: ``S = 244``, ``K/S = 41``, ``Z ~ 1.3``.

    References
    ----------
    Luby, M. (2002). LT codes. *Proc. 43rd IEEE FOCS*, 271-280.
    MacKay (2003), Section 50.2.

    Examples
    --------
    >>> r = robust_soliton(10000, 0.2, 0.05)
    >>> round(r.S, 6), r.m, round(r.Z, 6)
    (244.121453, 41, 1.31179)
    """
    if K < 2 or c <= 0 or not 0 < delta < 1:
        raise ValueError("need K >= 2, c > 0 and 0 < delta < 1")
    S = c * math.log(K / delta) * math.sqrt(K)
    m = int(math.floor(K / S + 0.5))
    if not 1 <= m <= K:
        raise ValueError("K / S must round to a degree in 1..K")
    rho = [1.0 / K] + [1.0 / (d * (d - 1)) for d in range(2, K + 1)]
    tau = [0.0] * K
    for d in range(1, m):
        tau[d - 1] = S / (K * d)
    tau[m - 1] = S * math.log(S / delta) / K
    Z = ssum(rho[i] + tau[i] for i in range(K))
    mu = [(rho[i] + tau[i]) / Z for i in range(K)]
    return RichResult(
        payload={
            "S": S,
            "m": m,
            "Z": Z,
            "rho": rho,
            "tau": tau,
            "mu": mu,
            "mean_degree": ssum((i + 1) * mu[i] for i in range(K)),
        }
    )


def code_length_decomposition(p, lengths) -> RichResult:
    r"""Expected length of a symbol code and its decomposition ``L = H(X) + D_KL(p || q) - log2 z``.

    With ``z = sum_i 2^{-l_i}`` (Kraft sum, ``z <= 1`` for a uniquely
    decodeable code) and implicit probabilities ``q_i = 2^{-l_i} / z``,
    ``L(C, X) = sum_i p_i l_i = sum_i p_i log2(1/q_i) - log2 z`` (MacKay eq. 5.14);
    for a complete code (``z = 1``) the excess over the entropy is exactly
    the relative entropy (eq. 5.23). All quantities in bits.

    Examples
    --------
    >>> r = code_length_decomposition([0.5, 0.25, 0.125, 0.125], [1, 2, 3, 3])
    >>> r.L, r.H, r.kl, r.kraft
    (1.75, 1.75, 0.0, 1.0)
    """
    p = [float(v) for v in p]
    ln = [float(v) for v in lengths]
    if len(p) != len(ln) or min(p) < 0:
        raise ValueError("p and lengths must match and p must be non-negative")
    tot = ssum(p)
    p = [v / tot for v in p]
    z = ssum(2.0**-v for v in ln)
    q = [2.0**-v / z for v in ln]
    L = ssum(p[i] * ln[i] for i in range(len(p)))
    H = ssum(-v * math.log2(v) for v in p if v > 0)
    kl = ssum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)
    return RichResult(payload={"L": L, "H": H, "kl": kl, "kraft": z, "q": q})


def _mi(pxy):
    px = [ssum(row) for row in pxy]
    py = [ssum(pxy[i][j] for i in range(len(pxy))) for j in range(len(pxy[0]))]
    return ssum(
        pxy[i][j] * math.log2(pxy[i][j] / (px[i] * py[j]))
        for i in range(len(pxy))
        for j in range(len(pxy[0]))
        if pxy[i][j] > 0
    )


def mcgill_interaction_information(pxyz) -> RichResult:
    r"""McGill's interaction information ``II = I(X; Y | Z) - I(X; Y)`` of a three-way pmf (bits).

    ``pxyz[i][j][k] = P(X = i, Y = j, Z = k)`` (normalised internally).
    ``II > 0`` means synergy (conditioning on Z raises the X-Y information),
    ``II < 0`` redundancy; ``II`` is symmetric in X, Y and Z.

    References
    ----------
    McGill, W. J. (1954). Multivariate information transmission. *Psychometrika*, 19, 97-116.

    Examples
    --------
    >>> xor = [[[0.25 if (i ^ j) == k else 0.0 for k in (0, 1)] for j in (0, 1)] for i in (0, 1)]
    >>> round(mcgill_interaction_information(xor).ii, 12)
    1.0
    """
    P = [[[float(v) for v in col] for col in row] for row in pxyz]
    tot = ssum(v for row in P for col in row for v in col)
    P = [[[v / tot for v in col] for col in row] for row in P]
    nx, ny, nz = len(P), len(P[0]), len(P[0][0])
    pxy = [[ssum(P[i][j]) for j in range(ny)] for i in range(nx)]
    ixy = _mi(pxy)
    icond = 0.0
    for k in range(nz):
        pz = ssum(P[i][j][k] for i in range(nx) for j in range(ny))
        if pz > 0:
            icond += pz * _mi([[P[i][j][k] / pz for j in range(ny)] for i in range(nx)])
    return RichResult(payload={"ii": icond - ixy, "i_xy": ixy, "i_xy_given_z": icond})


def cheatsheet() -> str:
    return (
        "bounded_distance_noise / self_dual_code_check / runlength_channel_capacity / repetition_error_approx / "
        "robust_soliton / code_length_decomposition / mcgill_interaction_information -> MacKay coding results."
    )


# alias kept from the retired placeholder of the same name
interaction_information = mcgill_interaction_information
