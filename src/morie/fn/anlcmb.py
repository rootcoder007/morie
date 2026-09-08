# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analytic combinatorics: coefficients, singularities and asymptotics.

Flajolet P, Sedgewick R (2009), *Analytic Combinatorics*, Cambridge
University Press -- Ch IV (rational and meromorphic), Ch VI
(singularity analysis). Hardy GH, Ramanujan S (1918) *Proc London Math
Soc* 17:75-115 (partition asymptotics). de Bruijn NG (1981),
*Asymptotic Methods in Analysis*, Dover (Stirling's series).

The discipline of the field is that an asymptotic estimate is a THEOREM
about exact coefficients, so every estimate here is computed alongside
the exact value it approximates and the two are compared. An asymptotic
formula that is never held against the sequence it describes is
decoration. Exact values are integers built by recurrence -- Python
integers are arbitrary-precision, so no coefficient here is ever
rounded -- and the estimates are doubles held to the error the theory
promises, not to the error that happens to come out.
"""

import math
from fractions import Fraction

from ._richresult import RichResult

__all__ = [
    "rational_gf_coefficients",
    "dominant_singularity_growth",
    "singularity_transfer",
    "stirling_series_error",
    "derangement_rounding",
    "hardy_ramanujan_partitions",
]

_METHOD = "Analytic combinatorics (Flajolet and Sedgewick 2009)"


def _partition_count_exact(n):
    """p(n) by Euler's pentagonal number recurrence, exact."""
    p = [1] + [0] * n
    for m in range(1, n + 1):
        total = 0
        k = 1
        while True:
            g1 = (k * (3 * k - 1)) // 2
            g2 = (k * (3 * k + 1)) // 2
            if g1 > m and g2 > m:
                break
            sign = 1 if k % 2 == 1 else -1
            if g1 <= m:
                total += sign * p[m - g1]
            if g2 <= m:
                total += sign * p[m - g2]
            k += 1
        p[m] = total
    return p


def rational_gf_coefficients(numerator, denominator, n_terms):
    r"""Power-series coefficients of a rational generating function.

    For :math:`P(x)/Q(x)` with :math:`Q(0) \ne 0`, the coefficients
    satisfy the linear recurrence read off :math:`Q`:

    .. math::
        a_n = \frac{1}{q_0}\Big(p_n - \sum_{i=1}^{\deg Q} q_i\,
        a_{n-i}\Big).

    Computed in exact rational arithmetic, so integer sequences come
    back as integers with nothing lost. Fibonacci is
    :math:`x/(1 - x - x^2)`, and this is checked in the tests against
    the recurrence directly.

    Parameters
    ----------
    numerator, denominator : sequences of coefficients, constant first.
    n_terms : number of coefficients to return.

    Examples
    --------
    >>> rational_gf_coefficients([0, 1], [1, -1, -1], 8)["coefficients"]
    [0, 1, 1, 2, 3, 5, 8, 13]
    >>> rational_gf_coefficients([1], [1, -2], 6)["coefficients"]
    [1, 2, 4, 8, 16, 32]
    """
    P = [Fraction(c) for c in numerator]
    Q = [Fraction(c) for c in denominator]
    m = int(n_terms)
    if not Q or Q[0] == 0:
        raise ValueError(
            "the denominator must have a non-zero constant term; a zero "
            "there is a pole at the origin, not a power series."
        )
    if m < 1:
        raise ValueError(f"n_terms must be positive; got {n_terms}.")
    coeffs = []
    for n in range(m):
        acc = P[n] if n < len(P) else Fraction(0)
        for i in range(1, min(n, len(Q) - 1) + 1):
            acc -= Q[i] * coeffs[n - i]
        coeffs.append(acc / Q[0])
    integral = all(c.denominator == 1 for c in coeffs)
    out = [int(c) if c.denominator == 1 else c for c in coeffs]
    return RichResult(
        title="Rational generating function coefficients",
        summary_lines=[
            ("Terms", m),
            ("All integral", integral),
        ],
        payload={
            "coefficients": out,
            "all_integral": integral,
            "estimate": float(out[-1]),
            "exact": str(out[-1]),
            "n": m,
            "method": _METHOD,
        },
    )


def dominant_singularity_growth(denominator, coefficients=None):
    r"""Exponential growth rate from the dominant singularity.

    The coefficients of a rational function grow like
    :math:`C\,\rho^{-n}` where :math:`\rho` is the root of the
    denominator closest to the origin (Flajolet-Sedgewick Theorem
    IV.7). The radius is found by bisection on the real line when the
    dominant root is real and positive -- the common combinatorial case,
    where coefficients are non-negative and Pringsheim's theorem puts a
    singularity on the positive real axis.

    If ``coefficients`` are supplied, the measured ratio
    :math:`a_{n+1}/a_n` of the last two is reported against the
    predicted :math:`1/\rho`, which is the check that the theorem is
    talking about THIS sequence.

    Examples
    --------
    >>> out = dominant_singularity_growth([1, -1, -1])
    >>> round(out["growth_rate"], 9)
    1.618033989
    """
    Q = [float(c) for c in denominator]
    if not Q or Q[0] == 0:
        raise ValueError("the denominator must have a non-zero constant "
                         "term.")

    def q(x):
        acc = 0.0
        for c in reversed(Q):
            acc = acc * x + c
        return acc

    # Pringsheim: scan out for the first sign change of Q on (0, hi]
    lo, hi = 0.0, 1e-6
    q0 = q(0.0)
    found = False
    while hi < 1e9:
        if q(hi) * q0 < 0:
            found = True
            break
        lo, hi = hi, hi * 2.0
    if not found:
        raise ValueError(
            "no positive real root of the denominator was found below "
            "1e9; the dominant singularity is complex or absent, and "
            "this routine only handles the Pringsheim case."
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if q(mid) * q0 < 0:
            hi = mid
        else:
            lo = mid
    rho = 0.5 * (lo + hi)
    rate = 1.0 / rho

    measured = None
    relative_gap = None
    if coefficients is not None:
        cs = list(coefficients)
        if len(cs) >= 2 and cs[-2] != 0:
            measured = float(cs[-1]) / float(cs[-2])
            relative_gap = abs(measured - rate) / rate
    out = RichResult(
        title="Dominant singularity",
        summary_lines=[
            ("Radius of convergence", rho),
            ("Growth rate 1/rho", rate),
            ("Measured a(n+1)/a(n)", measured),
        ],
        payload={
            "radius": rho,
            "growth_rate": rate,
            "estimate": rate,
            "measured_ratio": measured,
            "relative_gap": relative_gap,
            "n": len(Q) - 1,
            "method": _METHOD,
        },
    )
    return out


def singularity_transfer(alpha, n):
    r"""The basic transfer theorem: coefficients of
    :math:`(1-x)^{-\alpha}`.

    Exactly, :math:`[x^n](1-x)^{-\alpha} = \binom{n+\alpha-1}{n}
    = \prod_{i=1}^{n}\frac{\alpha + i - 1}{i}`; asymptotically it is
    :math:`n^{\alpha-1}/\Gamma(\alpha)` (Flajolet-Sedgewick Theorem
    VI.1). Both are computed, with the known first-order correction
    :math:`1 + \alpha(\alpha-1)/(2n)`, and the ratio is reported.
    This is the engine behind the :math:`4^n/\sqrt{\pi n^3}` shape of
    the Catalan numbers, which the tests derive from it rather than
    restate.

    Examples
    --------
    >>> out = singularity_transfer(0.5, 100)
    >>> round(out["corrected_ratio"], 5)
    1.0
    """
    a = float(alpha)
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    if a <= 0 and a == int(a):
        raise ValueError(
            f"alpha must not be a non-positive integer; got {alpha}: "
            "(1-x)^-alpha is a polynomial there and the transfer "
            "theorem does not apply."
        )
    exact = 1.0
    for i in range(1, n + 1):
        exact *= (a + i - 1) / i
    asym = n ** (a - 1.0) / math.gamma(a)
    corrected = asym * (1.0 + a * (a - 1.0) / (2.0 * n))
    return RichResult(
        title=f"Transfer theorem at alpha = {a}",
        summary_lines=[
            ("Exact coefficient", exact),
            ("n^(a-1)/Gamma(a)", asym),
            ("Ratio exact/asymptotic", exact / asym),
        ],
        payload={
            "exact_coefficient": exact,
            "asymptotic": asym,
            "corrected": corrected,
            "ratio": exact / asym,
            "corrected_ratio": exact / corrected,
            "estimate": asym,
            "alpha": a,
            "n": n,
            "method": _METHOD,
        },
    )


def stirling_series_error(n, terms=3):
    r"""Stirling's series for :math:`\ln n!`, with its error bound
    checked.

    .. math::
        \ln n! = n\ln n - n + \tfrac{1}{2}\ln(2\pi n)
        + \sum_{k=1}^{K} \frac{B_{2k}}{2k(2k-1)n^{2k-1}} + R_K,

    and because the series is alternating-enveloping, :math:`|R_K|` is
    at most the first omitted term (de Bruijn 1981, section 3.10). The
    truth is :math:`\ln\Gamma(n+1)`, and both the achieved error and
    the promised bound are returned so the promise can be tested rather
    than trusted.

    The promise outruns the arithmetic quickly: at :math:`n = 50` with
    three terms the bound is :math:`7.6 \times 10^{-16}` while
    :math:`\ln 50! \approx 148`, whose representable neighbours are
    :math:`3 \times 10^{-14}` apart. Below that resolution the check
    would be measuring the rounding of the comparison, not the series,
    so ``error_within_bound`` is judged against the bound plus the
    double-precision floor, which is reported separately as
    ``double_floor``.

    Examples
    --------
    >>> out = stirling_series_error(10)
    >>> out["error_within_bound"]
    True
    """
    n = int(n)
    K = int(terms)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    if not 0 <= K <= 4:
        raise ValueError(
            f"terms must lie in 0..4; got {terms}: the error bound is "
            "the first OMITTED term, so K = 5 would need B12, which is "
            "not tabulated here. A bound faked from a lower Bernoulli "
            "number would be a promise the theory never made."
        )
    # B2, B4, B6, B8, B10
    bern = [Fraction(1, 6), Fraction(-1, 30), Fraction(1, 42),
            Fraction(-1, 30), Fraction(5, 66)]
    base = n * math.log(n) - n + 0.5 * math.log(2.0 * math.pi * n)
    approx = base
    for k in range(1, K + 1):
        b = bern[k - 1]
        approx += float(b) / (2 * k * (2 * k - 1) * n ** (2 * k - 1))
    kk = K + 1
    bound = abs(float(bern[kk - 1])) / (2 * kk * (2 * kk - 1)
                                        * n ** (2 * kk - 1))
    truth = math.lgamma(n + 1)
    err = abs(approx - truth)
    # the series bound falls below double resolution quickly -- at
    # n = 50 with three terms it is 7.6e-16 while ln 50! is about 148,
    # whose representable neighbours are 3e-14 apart. The achieved
    # error can never be judged below that floor, so the check is
    # against bound + floor and the floor is reported separately.
    floor = 8.0 * abs(truth) * 2.0 ** -53
    return RichResult(
        title=f"Stirling series at n = {n}, {K} correction terms",
        summary_lines=[
            ("ln n! (truth)", truth),
            ("Series value", approx),
            ("Achieved error", err),
            ("Promised bound", bound),
        ],
        payload={
            "log_factorial": truth,
            "series_value": approx,
            "error": err,
            "bound": bound,
            "double_floor": floor,
            "error_within_bound": err <= bound + floor,
            "estimate": approx,
            "terms": K,
            "n": n,
            "method": "Stirling's series (de Bruijn 1981, section 3.10)",
        },
    )


def derangement_rounding(n):
    r"""The rounding identity :math:`D_n = \mathrm{round}(n!/e)`.

    The inclusion-exclusion sum gives
    :math:`|D_n - n!/e| = |\sum_{k>n} (-1)^k n!/k!| < 1/(n+1)`, so for
    :math:`n \ge 1` the derangement number is the NEAREST INTEGER to
    :math:`n!/e`. Checked exactly: :math:`D_n` by the recurrence
    :math:`D_n = (n-1)(D_{n-1} + D_{n-2})` in exact integers, the
    distance :math:`|D_n \cdot e - n!|/e` bounded through a rational
    enclosure of :math:`e` from the tail of its own series. No doubles
    are involved -- past :math:`n = 18`, :math:`n!/e` does not fit one.

    Examples
    --------
    >>> derangement_rounding(4)["derangements"]
    9
    >>> derangement_rounding(10)["is_nearest_integer"]
    True
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}.")
    D = [1, 0]
    for m in range(2, n + 1):
        D.append((m - 1) * (D[m - 1] + D[m - 2]))
    dn = D[n] if n >= 1 else 1
    fact = math.factorial(n)
    # rational enclosure of e: sum_{k<=K} 1/k! < e < same + 2/(K+1)!
    K = max(n + 3, 20)
    lo = sum(Fraction(1, math.factorial(k)) for k in range(K + 1))
    hi = lo + Fraction(2, math.factorial(K + 1))
    # distance |D_n - n!/e| enclosed by [n!/hi, n!/lo] around D_n
    d_lo = abs(Fraction(dn) - Fraction(fact) / lo)
    d_hi = abs(Fraction(dn) - Fraction(fact) / hi)
    dist_max = max(d_lo, d_hi)
    nearest = dist_max < Fraction(1, 2)
    within = dist_max < Fraction(1, n + 1) if n >= 1 else True
    return RichResult(
        title=f"Derangements of {n}",
        summary_lines=[
            ("D_n", dn),
            ("Nearest integer to n!/e", nearest),
            ("Distance below 1/(n+1)", within),
        ],
        payload={
            "derangements": dn,
            "exact": str(dn),
            "estimate": float(dn),
            "factorial": fact,
            "distance_bound": float(dist_max),
            "is_nearest_integer": nearest,
            "within_theoretical_bound": within,
            "n": n,
            "method": "Meromorphic asymptotics: D_n = round(n!/e)",
        },
    )


def hardy_ramanujan_partitions(n):
    r"""Hardy-Ramanujan asymptotics for the partition function.

    .. math::
        p(n) \sim \frac{1}{4n\sqrt{3}}
        \exp\!\Big(\pi\sqrt{\tfrac{2n}{3}}\Big),

    against :math:`p(n)` computed exactly by Euler's pentagonal
    recurrence. Convergence is famously slow -- the relative error decays
    like :math:`n^{-1/2}`, so at :math:`n = 100` the estimate is still
    4.6 per cent high -- and the point of returning both is that the
    formula's fame should not be mistaken for accuracy at small
    :math:`n`.

    Examples
    --------
    >>> hardy_ramanujan_partitions(100)["partitions"]
    190569292
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    exact = _partition_count_exact(n)[n]
    asym = math.exp(math.pi * math.sqrt(2.0 * n / 3.0)) / (4.0 * n
                                                           * math.sqrt(3.0))
    return RichResult(
        title=f"Partitions of {n}",
        summary_lines=[
            ("p(n) exact", exact),
            ("Hardy-Ramanujan", asym),
            ("Ratio asymptotic/exact", asym / exact),
        ],
        payload={
            "partitions": exact,
            "exact": str(exact),
            "estimate": asym,
            "asymptotic": asym,
            "ratio": asym / exact,
            "relative_error": asym / exact - 1.0,
            "n": n,
            "method": "Hardy-Ramanujan asymptotic (1918)",
        },
    )


def cheatsheet():
    return (
        "anlcmb: rational generating functions expanded exactly, growth "
        "rates from the dominant singularity, the transfer theorem, "
        "Stirling's series with its error bound checked, the n!/e rounding "
        "identity for derangements proved in rational arithmetic, and "
        "Hardy-Ramanujan held against exact p(n)"
    )
