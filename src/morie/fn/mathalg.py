# morie.fn -- function file (rootcoder007/morie)
"""Numerical and discrete algorithms: Pollard's rho factorisation, Legendre polynomial bases by the
three-term recurrence, propositional resolution refutation, the recursive polar transform and the
Panter-Dite distortion of high-rate scalar quantisation, the logit transform of proportions for
meta-analysis, and Monte Carlo standard errors from Geyer's initial sequence."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import ssum
from ._richresult import RichResult

__all__ = [
    "pollard_rho",
    "legendre_polynomials",
    "resolution_refutation",
    "polar_transform",
    "panter_dite_bound",
    "logit_proportion",
    "mc_standard_error",
]

_LIMIT = 2**52


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _rho(n, c):
    x = y = 2
    d = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
    return d


def pollard_rho(n):
    r"""Prime factorisation by trial division of small primes and Pollard's rho with Floyd cycle detection.

    For composite ``n`` the sequence ``x -> x^2 + c (mod n)`` from ``x = 2`` is
    run with a tortoise and a hare; ``gcd(|x - y|, n)`` reveals a factor once
    the sequence cycles modulo it (expected ``O(n^(1/4))`` steps). On failure
    (``d = n``) ``c`` is increased. Primality uses the deterministic
    Miller-Rabin test; ``n < 2^52`` so that the R arm stays exact.

    References
    ----------
    Pollard, J. M. (1975). A Monte Carlo method for factorization. *BIT* 15,
    331-334.

    Examples
    --------
    >>> pollard_rho(8051)
    [83, 97]
    >>> pollard_rho(600851475143)
    [71, 839, 1471, 6857]
    """
    n = int(n)
    if n < 2 or n >= _LIMIT:
        raise ValueError("n must satisfy 2 <= n < 2^52")
    out = []
    for p in (2, 3, 5, 7, 11, 13):
        while n % p == 0:
            out.append(p)
            n //= p
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if _is_prime(m):
            out.append(m)
            continue
        c = 1
        d = _rho(m, c)
        while d == m:
            c += 1
            d = _rho(m, c)
        stack += [d, m // d]
    return sorted(out)


def legendre_polynomials(x, degree, normalized=False):
    r"""Legendre polynomials ``P_0 .. P_K`` at ``x`` by Bonnet's recurrence.

    ``(k + 1) P_{k+1}(x) = (2k + 1) x P_k(x) - k P_{k-1}(x)``, ``P_0 = 1``,
    ``P_1 = x``; with ``normalized`` each is scaled by ``sqrt((2k + 1)/2)`` to
    be orthonormal on [-1, 1]. Returns one row per ``x`` value.

    References
    ----------
    Abramowitz, M. and Stegun, I. A. (1964). *Handbook of Mathematical
    Functions*, 22.7.10.

    Examples
    --------
    >>> legendre_polynomials([0.5], 3)
    [[1.0, 0.5, -0.125, -0.4375]]
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]
    K = int(degree)
    out = []
    for v in xs:
        p = [1.0, v][: K + 1]
        for k in range(1, K):
            p.append(((2 * k + 1) * v * p[k] - k * p[k - 1]) / (k + 1))
        if normalized:
            p = [q * math.sqrt((2 * k + 1) / 2.0) for k, q in enumerate(p)]
        out.append(p)
    return out


def resolution_refutation(clauses, max_clauses=100000):
    r"""Propositional resolution: saturate a clause set and report unsatisfiability.

    Clauses are lists of non-zero integers (DIMACS literals: ``-k`` is the
    negation of variable ``k``). Pairs of clauses with a complementary literal
    are resolved (tautologies discarded) in a fixed breadth-first order until
    the empty clause appears (unsatisfiable, with the derivation returned) or
    no new clause can be produced (satisfiable).

    References
    ----------
    Robinson, J. A. (1965). A machine-oriented logic based on the resolution
    principle. *Journal of the ACM* 12, 23-41.

    Examples
    --------
    >>> r = resolution_refutation([[1, 2], [-1, 2], [1, -2], [-1, -2]])
    >>> r.unsatisfiable, len(r.clauses) > 4
    (True, True)
    """
    cl = []
    seen = set()
    for c in clauses:
        k = tuple(sorted(set(int(v) for v in c)))
        if k not in seen and not any(-v in k for v in k):
            seen.add(k)
            cl.append(k)
    parents = [None] * len(cl)
    i = 0
    while i < len(cl):
        for j in range(i):
            a, b = cl[j], cl[i]
            for lit in a:
                if -lit in b:
                    r = tuple(sorted(set([v for v in a if v != lit] + [v for v in b if v != -lit])))
                    if any(-v in r for v in r) or r in seen:
                        continue
                    seen.add(r)
                    cl.append(r)
                    parents.append((j, i, abs(lit)))
                    if not r:
                        return RichResult(
                            payload={"unsatisfiable": True, "clauses": [list(c) for c in cl], "parents": parents}
                        )
                    if len(cl) > max_clauses:
                        raise RuntimeError("clause limit reached")
        i += 1
    return RichResult(payload={"unsatisfiable": False, "clauses": [list(c) for c in cl], "parents": parents})


def polar_transform(x, inverse=False):
    r"""Recursive polar transform of a vector of length ``2^L`` (PolarQuant).

    Level 1 maps each coordinate pair ``(x_{2i}, x_{2i+1})`` to a radius and the
    angle ``atan2(x_{2i+1}, x_{2i})``; each further level maps pairs of radii
    to a radius and ``atan2(r_{2i+1}, r_{2i})`` (in [0, pi/2]); the last radius is
    ``||x||_2``. ``inverse=True`` takes ``(radius, angles)`` (angles listed
    level by level) back to ``x``.

    References
    ----------
    Han, I., Kacham, P., Karbasi, A., Mirrokni, V. and Zandieh, A. (2025).
    PolarQuant: quantizing KV caches with polar transformation. arXiv
    2502.02617.

    Examples
    --------
    >>> r = polar_transform([3.0, 4.0, 0.0, 0.0])
    >>> r.radius, [round(a, 12) for a in r.angles]
    (5.0, [0.927295218002, 0.0, 0.0])
    >>> [round(v, 12) + 0.0 for v in polar_transform((r.radius, r.angles), inverse=True)]
    [3.0, 4.0, 0.0, 0.0]
    """
    if not inverse:
        v = [float(t) for t in np.asarray(x, dtype=float).ravel().tolist()]
        d = len(v)
        if d < 2 or d & (d - 1):
            raise ValueError("length must be a power of 2, at least 2")
        angles = []
        while len(v) > 1:
            r = []
            for i in range(0, len(v), 2):
                r.append(math.hypot(v[i], v[i + 1]))
                angles.append(math.atan2(v[i + 1], v[i]))
            v = r
        return RichResult(payload={"radius": v[0], "angles": angles})
    radius, angles = x
    ang = [float(a) for a in angles]
    d = len(ang) + 1
    sizes = []
    m = d // 2
    while m >= 1:
        sizes.append(m)
        m //= 2
    blocks, pos = [], 0
    for s in sizes:
        blocks.append(ang[pos : pos + s])
        pos += s
    v = [float(radius)]
    for blk in reversed(blocks):
        v = [c for r, a in zip(v, blk) for c in (r * math.cos(a), r * math.sin(a))]
    return v


def panter_dite_bound(bits, sigma2=1.0):
    r"""Panter-Dite high-resolution distortion of optimal scalar quantisation of a Gaussian source.

    ``D(b) = (sqrt(3) pi / 2) sigma^2 2^{-2b}`` (the Panter and Dite 1951
    integral ``(1/12) (int p^{1/3})^3 2^{-2b}`` for a normal density), which
    the Lloyd-Max quantiser approaches as ``b`` grows; also returned is the
    corresponding SNR in dB.

    References
    ----------
    Panter, P. F. and Dite, W. (1951). Quantization distortion in pulse-count
    modulation with nonuniform spacing of levels. *Proceedings of the IRE*
    39, 44-48.

    Examples
    --------
    >>> r = panter_dite_bound(4)
    >>> round(r.mse, 12)
    0.01062773065
    """
    mse = math.sqrt(3.0) * math.pi / 2.0 * sigma2 * 2.0 ** (-2 * bits)
    return RichResult(payload={"mse": mse, "snr_db": 10 * math.log10(sigma2 / mse)})


def logit_proportion(events, n):
    r"""Logit-transformed proportions and their variances for meta-analysis.

    ``y = log(p / (1 - p))`` with ``p = x / n`` and ``v = 1 / (n p) + 1 / (n (1 -
    p)) = 1/x + 1/(n - x)`` (``metafor::escalc(measure = "PLO")``);
    back-transform with the inverse logit.

    References
    ----------
    Lipsey, M. W. and Wilson, D. B. (2001). *Practical Meta-Analysis*. Sage.

    Examples
    --------
    >>> r = logit_proportion([12, 30], [40, 50])
    >>> [round(v, 12) for v in r.yi], [round(v, 12) for v in r.vi]
    ([-0.847297860387, 0.405465108108], [0.119047619048, 0.083333333333])
    """
    x = [float(v) for v in np.asarray(events, dtype=float).ravel().tolist()]
    m = [float(v) for v in np.asarray(n, dtype=float).ravel().tolist()]
    yi = [math.log(a / (b - a)) for a, b in zip(x, m)]
    vi = [1.0 / a + 1.0 / (b - a) for a, b in zip(x, m)]
    return RichResult(payload={"yi": yi, "vi": vi})


def mc_standard_error(draws):
    r"""Monte Carlo standard error of a chain mean by Geyer's initial sequence estimators.

    With autocovariances ``gamma_k = (1/n) sum_j x_j x_{j+k}`` of the centred
    chain and ``Gamma_i = gamma_{2i} + gamma_{2i+1}`` kept while positive, the
    initial positive (``var_pos``) and initial monotone (``var_dec``)
    estimates of the asymptotic variance are ``-gamma_0 + 2 sum Gamma_i``
    (the monotone one after replacing each ``Gamma_i`` by the running
    minimum); ``mcse = sqrt(var_dec / n)`` and the effective sample size ``n
    gamma_0 / var_dec``, so ``mcse = sd / sqrt(ESS)``. Equal to
    ``mcmc::initseq``.

    References
    ----------
    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
    Science* 7, 473-483.

    Examples
    --------
    >>> r = mc_standard_error([0.1, 0.4, 0.3, 0.8, 0.6, 0.2, 0.5, 0.9])
    >>> round(r.mcse, 12)
    0.092174257523
    """
    v = [float(t) for t in np.asarray(draws, dtype=float).ravel().tolist()]
    n = len(v)
    mu = ssum(v) / n
    x = [t - mu for t in v]

    def gam(k):
        return ssum(x[j] * x[j + k] for j in range(n - k)) / n

    g0 = gam(0)
    big = []
    for i in range(n // 2):
        g = gam(2 * i) + gam(2 * i + 1)
        if g <= 0:
            break
        big.append(g)
    var_pos = -g0 + 2 * ssum(big)
    for i in range(1, len(big)):
        if big[i] > big[i - 1]:
            big[i] = big[i - 1]
    var_dec = -g0 + 2 * ssum(big)
    return RichResult(
        payload={
            "mcse": math.sqrt(var_dec / n),
            "ess": n * g0 / var_dec,
            "gamma0": g0,
            "var_pos": var_pos,
            "var_dec": var_dec,
        }
    )


def cheatsheet() -> str:
    return (
        "pollard_rho / legendre_polynomials / resolution_refutation / polar_transform / panter_dite_bound / "
        "logit_proportion / mc_standard_error -> numerical and discrete algorithms."
    )


# alias kept from the retired placeholder of the same name
legendre_basis = legendre_polynomials

# alias kept from the retired placeholder of the same name
ma_logit_transform = logit_proportion

# alias kept from the retired placeholder of the same name
mcmc_standard_error = mc_standard_error

# alias kept from the retired placeholder of the same name
pollards_rho = pollard_rho

# alias kept from the retired placeholder of the same name
resolution_proof = resolution_refutation

# alias kept from the retired placeholder of the same name
turboquant_mse_distortion_bound = panter_dite_bound

# alias kept from the retired placeholder of the same name
turboquant_polar_transform = polar_transform
