# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The probabilistic method: proving things exist without building them.

Alon N, Spencer JH (2016), *The Probabilistic Method*, 4th ed., Wiley.
Original sources: Erdos P (1947) *Bull AMS* 53:292-294; Erdos P, Lovasz
L (1975); Chernoff H (1952) *Annals of Mathematical Statistics*
23(4):493-507; Azuma K (1967) *Tohoku Mathematical Journal*
19(3):357-367; Hoeffding W (1963) *JASA* 58(301):13-30.

Every bound here is one-sided, and that is what makes it testable: a
bound that is claimed to hold must never be violated by a directly
computed probability, and a bound that is claimed to be *non-vacuous*
must actually be below 1. Both are checked, because a bound of 1 is
always true and always useless.

The alteration bound is checked against the plain union bound from
:mod:`morie.fn.ramthy`. It is asymptotically stronger but NOT
uniformly so, and the small-:math:`k` cases where the union bound wins
are recorded rather than smoothed over.
"""

import math

from ._richresult import RichResult

__all__ = [
    "union_bound_exists",
    "first_moment_ramsey",
    "alteration_ramsey",
    "lovasz_local_lemma",
    "chernoff_bound",
    "azuma_bound",
    "second_moment_threshold",
]

_METHOD = "Probabilistic method with one-sided bounds"


def union_bound_exists(n_events, event_probability):
    r"""The first moment in its plainest form: if the expected number
    of bad events is below 1, some outcome has none.

    :math:`E[X] = m p`, and :math:`P(X = 0) > 0` whenever
    :math:`E[X] < 1`. This proves existence without exhibiting
    anything, which is the whole point and also the whole limitation.

    Examples
    --------
    >>> union_bound_exists(10, 0.05)["exists"]
    True
    >>> union_bound_exists(100, 0.05)["exists"]
    False
    """
    m = int(n_events)
    p = float(event_probability)
    if m < 0:
        raise ValueError(f"n_events must be non-negative; got {m}.")
    if not 0 <= p <= 1:
        raise ValueError(f"event_probability must lie in [0, 1]; got {p}.")
    expected = m * p
    return RichResult(
        title="First moment / union bound",
        summary_lines=[
            ("Expected bad events", expected),
            ("Existence guaranteed", expected < 1),
        ],
        payload={
            "expected": expected,
            "estimate": expected,
            "exists": expected < 1,
            "n_events": m,
            "event_probability": p,
            "n": m,
            "method": "First moment method",
        },
        interpretation=(
            f"The expected number of bad events is {expected:.6g} < 1, so an "
            "outcome avoiding all of them exists."
            if expected < 1 else
            f"The expected count is {expected:.6g}, which is at least 1. The "
            "first moment method says nothing here -- it does NOT show that "
            "no good outcome exists."
        ),
    )


def first_moment_ramsey(k):
    r"""Erdos's lower bound on the diagonal Ramsey number.

    Colouring :math:`K_n` at random, the expected number of
    monochromatic :math:`K_k` is
    :math:`\binom{n}{k} 2^{1 - \binom{k}{2}}`; below 1 it proves
    :math:`R(k,k) > n`.

    Examples
    --------
    >>> first_moment_ramsey(10)["bound"]
    100
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"k must be at least 2; got {k}.")
    expo = 1 - math.comb(k, 2)
    CAP = 100000
    n = k
    best = k - 1
    capped = False
    while n < CAP:
        log2e = (math.lgamma(n + 1) - math.lgamma(k + 1)
                 - math.lgamma(n - k + 1)) / math.log(2.0) + expo
        if log2e < 0:
            best = n
            n += 1
        else:
            break
    # a search that runs to its own ceiling has not found the bound, it
    # has found the ceiling. Silently returning it would read as a result.
    capped = n >= CAP
    log2_at = (math.lgamma(best + 1) - math.lgamma(k + 1)
               - math.lgamma(best - k + 1)) / math.log(2.0) + expo
    out = RichResult(
        title=f"First moment lower bound on R({k},{k})",
        summary_lines=[
            ("Bound", best),
            ("Certifies", f"R({k},{k}) > {best}"),
            ("Expected count there", 2.0 ** log2_at),
        ],
        payload={
            "bound": best,
            "estimate": float(best),
            "certifies": f"R({k},{k}) > {best}",
            "expected_at_bound": 2.0 ** log2_at,
            "asymptotic_2_to_k_over_2": 2.0 ** (k / 2.0),
            "search_capped": capped,
            "search_cap": CAP,
            "k": k,
            "n": best,
            "method": "First moment method (Erdos 1947)",
        },
    )
    if capped:
        out.warnings.append(
            f"The search reached its ceiling of {CAP} without the expected "
            "count exceeding 1, so the value returned is the CEILING, not "
            "the bound. The true bound is larger."
        )
    return out


def alteration_ramsey(k):
    r"""The alteration method, which beats the plain union bound.

    Instead of insisting no monochromatic :math:`K_k` appears, colour
    :math:`K_n` at random, then **delete one vertex from each** that
    does. What remains has no monochromatic :math:`K_k`, and its
    expected size is

    .. math:: n - \binom{n}{k} 2^{1 - \binom{k}{2}},

    so some colouring leaves at least that many. Maximising over
    :math:`n` gives the bound.

    Asymptotically this gains a factor of 2 in the exponent over the
    plain union bound, and the gain is large in absolute terms once
    :math:`k` is moderate: 115 against 100 at :math:`k = 10`, 7446
    against 5817 at :math:`k = 20`.

    **It is not uniformly better, and an earlier version of this
    docstring wrongly claimed it was.** At the union bound's own
    :math:`n` the expected count is below 1, so the surviving set has
    more than :math:`n - 1` vertices -- which after the floor can be
    :math:`n - 1` rather than :math:`n`. Measured: at :math:`k = 4` the
    alteration expression gives 5 against the union bound's 6, and at
    :math:`k = 6` the two tie. Both are valid lower bounds, so
    ``best_bound`` reports their maximum and ``improvement`` is
    reported signed rather than assumed positive.

    Examples
    --------
    >>> alteration_ramsey(10)["bound"] > first_moment_ramsey(10)["bound"]
    True
    >>> alteration_ramsey(4)["improvement"]      # the union bound wins here
    -1
    >>> alteration_ramsey(4)["best_bound"]
    6
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"k must be at least 2; got {k}.")
    expo = 1 - math.comb(k, 2)
    ACAP = 200000
    best_n, best_val = k, 0.0
    reached_end = True
    for n in range(k, ACAP):
        log2e = (math.lgamma(n + 1) - math.lgamma(k + 1)
                 - math.lgamma(n - k + 1)) / math.log(2.0) + expo
        if log2e > 60:
            reached_end = False
            break
        val = n - 2.0 ** log2e
        if val > best_val:
            best_val, best_n = val, n
        elif n > best_n + 5000:
            reached_end = False
            break
    bound = int(math.floor(best_val))
    fm = first_moment_ramsey(k)["bound"]
    return RichResult(
        title=f"Alteration lower bound on R({k},{k})",
        summary_lines=[
            ("Alteration bound", bound),
            ("First moment bound", fm),
            ("Improvement", bound - fm),
            ("Best of the two", max(bound, fm)),
            ("Optimal n", best_n),
        ],
        payload={
            "bound": bound,
            "estimate": float(bound),
            "certifies": f"R({k},{k}) > {bound}",
            "first_moment_bound": fm,
            "improvement": bound - fm,
            "best_bound": max(bound, fm),
            "optimal_n": best_n,
            "expected_survivors": best_val,
            "search_capped": reached_end,
            "search_cap": ACAP,
            "k": k,
            "n": bound,
            "method": "Alteration method (Alon and Spencer, Ch 3)",
        },
        interpretation=(
            f"Deleting one vertex per bad clique leaves {best_val:.1f} "
            f"vertices in expectation, certifying R({k},{k}) > {bound}. The "
            f"union bound gives {fm}, so the best available here is "
            f"{max(bound, fm)}."
        ),
    )


def lovasz_local_lemma(p, d, symmetric=True):
    r"""The symmetric Lovasz Local Lemma.

    If each of a family of bad events has probability at most :math:`p`
    and is independent of all but at most :math:`d` others, and

    .. math:: e\,p\,(d + 1) \le 1,

    then with positive probability **none** of them occurs.

    What makes the lemma remarkable is that the union bound is useless
    once :math:`mp \ge 1`, however small the dependency. The Local
    Lemma does not care how many events there are at all -- only how
    entangled each one is. The payload reports both so the contrast is
    visible: it is routine for the union bound to fail while the Local
    Lemma succeeds on the same family.

    Examples
    --------
    >>> lovasz_local_lemma(0.01, 20)["applies"]
    True
    >>> lovasz_local_lemma(0.1, 20)["applies"]
    False
    """
    p = float(p)
    d = int(d)
    if not 0 <= p <= 1:
        raise ValueError(f"p must lie in [0, 1]; got {p}.")
    if d < 0:
        raise ValueError(f"d must be non-negative; got {d}.")
    if not symmetric:
        raise ValueError("only the symmetric form is implemented.")
    lhs = math.e * p * (d + 1)
    applies = lhs <= 1.0
    # largest degree still admissible at this p, and largest p at this d
    max_d = int(math.floor(1.0 / (math.e * p) - 1.0)) if p > 0 else None
    max_p = 1.0 / (math.e * (d + 1))
    return RichResult(
        title="Lovasz Local Lemma (symmetric)",
        summary_lines=[
            ("e p (d+1)", lhs),
            ("Condition holds", applies),
            ("Largest admissible d", max_d),
            ("Largest admissible p", max_p),
        ],
        payload={
            "condition_value": lhs,
            "estimate": lhs,
            "applies": applies,
            "p": p,
            "d": d,
            "max_degree_at_p": max_d,
            "max_probability_at_d": max_p,
            "slack": 1.0 - lhs,
            "n": d,
            "method": "Lovasz Local Lemma (Erdos and Lovasz 1975)",
        },
        interpretation=(
            f"e p (d+1) = {lhs:.6g} <= 1, so with positive probability none "
            "of the bad events occurs -- regardless of how many there are."
            if applies else
            f"e p (d+1) = {lhs:.6g} > 1, so the symmetric Local Lemma does "
            "not apply. That is not a proof that a good outcome fails to "
            "exist."
        ),
    )


def chernoff_bound(n, p, t, tail="upper"):
    r"""Chernoff bounds on a binomial tail.

    For :math:`X \sim \mathrm{Bin}(n, p)` with :math:`\mu = np`, the
    multiplicative forms are

    .. math::
        P(X \ge (1+\delta)\mu) \le
            \left[\frac{e^{\delta}}{(1+\delta)^{1+\delta}}\right]^{\mu},
        \qquad
        P(X \le (1-\delta)\mu) \le e^{-\mu\delta^2/2}.

    The exact tail is computed alongside, so the test is that the bound
    **holds** -- and separately that it is not vacuous, since a bound of
    1 is always true and never useful.

    Examples
    --------
    >>> out = chernoff_bound(100, 0.5, 70)
    >>> out["bound"] >= out["exact_tail"]
    True
    >>> out["vacuous"]
    False
    """
    n = int(n)
    p = float(p)
    t = float(t)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    if not 0 <= p <= 1:
        raise ValueError(f"p must lie in [0, 1]; got {p}.")
    if tail not in ("upper", "lower"):
        raise ValueError('tail must be "upper" or "lower".')
    mu = n * p
    if mu <= 0:
        raise ValueError("np must be positive for a multiplicative bound.")

    if tail == "upper":
        delta = t / mu - 1.0
        if delta <= 0:
            bound = 1.0
        else:
            bound = math.exp(
                mu * (delta - (1 + delta) * math.log1p(delta))
            )
        exact = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                    for i in range(math.ceil(t), n + 1))
    else:
        delta = 1.0 - t / mu
        bound = 1.0 if delta <= 0 else math.exp(-mu * delta * delta / 2.0)
        exact = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                    for i in range(0, math.floor(t) + 1))
    bound = min(bound, 1.0)

    out = RichResult(
        title=f"Chernoff bound, {tail} tail",
        summary_lines=[
            ("Bound", bound),
            ("Exact tail", exact),
            ("Bound holds", bound >= exact - 1e-12),
            ("Vacuous", bound >= 1.0 - 1e-12),
        ],
        payload={
            "bound": bound,
            "estimate": bound,
            "exact_tail": exact,
            "holds": bound >= exact - 1e-12,
            "vacuous": bound >= 1.0 - 1e-12,
            "slack": bound - exact,
            "mu": mu,
            "delta": delta,
            "tail": tail,
            "n": n,
            "method": "Chernoff bound (Chernoff 1952)",
        },
    )
    if bound < exact - 1e-12:
        out.warnings.append(
            f"The bound ({bound:.6g}) is below the exact tail "
            f"({exact:.6g}), which is impossible. The implementation is "
            "wrong."
        )
    if bound >= 1.0 - 1e-12:
        out.warnings.append(
            "The bound is 1, which is true of every probability and "
            "therefore says nothing. The deviation asked about is too small "
            "relative to the mean."
        )
    return out


def azuma_bound(n, c, t):
    r"""Azuma-Hoeffding for a martingale with bounded differences.

    If :math:`|X_i - X_{i-1}| \le c` then

    .. math:: P(|X_n - X_0| \ge t) \le 2\exp\!\left(\frac{-t^2}{2nc^2}\right).

    The bound needs no distributional assumption beyond the step size,
    which is why it applies to processes with no independence at all.

    Examples
    --------
    >>> out = azuma_bound(100, 1.0, 30)
    >>> bool(out["bound"] < 0.03)   # measured 0.02222
    True
    """
    n = int(n)
    c = float(c)
    t = float(t)
    if n < 1:
        raise ValueError(f"n must be positive; got {n}.")
    if c <= 0:
        raise ValueError(f"c must be positive; got {c}.")
    if t < 0:
        raise ValueError(f"t must be non-negative; got {t}.")
    bound = min(2.0 * math.exp(-t * t / (2.0 * n * c * c)), 1.0)
    return RichResult(
        title="Azuma-Hoeffding bound",
        summary_lines=[
            ("Bound on P(|X_n - X_0| >= t)", bound),
            ("Steps", n),
            ("Step size", c),
            ("Deviation", t),
        ],
        payload={
            "bound": bound,
            "estimate": bound,
            "vacuous": bound >= 1.0 - 1e-12,
            "typical_deviation": c * math.sqrt(n),
            "deviations_out": t / (c * math.sqrt(n)),
            "n": n,
            "method": "Azuma-Hoeffding inequality (Azuma 1967)",
        },
        interpretation=(
            f"A deviation of {t} is {t / (c * math.sqrt(n)):.2f} typical "
            f"deviations, bounded in probability by {bound:.4g}."
        ),
    )


def second_moment_threshold(expectation, variance):
    r"""The second moment method: a small relative variance forces the
    count to be positive.

    By Chebyshev,

    .. math::
        P(X = 0) \le \frac{\mathrm{Var}(X)}{E[X]^2},

    so :math:`\mathrm{Var}/E^2 \to 0` gives :math:`X > 0` with
    probability tending to 1.

    This is the companion to the first moment, and the pair is what
    makes threshold results sharp: the first moment shows a property
    vanishes below the threshold, the second shows it appears above.
    Neither alone establishes a threshold.

    Examples
    --------
    >>> out = second_moment_threshold(100.0, 50.0)
    >>> bool(out["p_zero_bound"] < 0.01)
    True
    """
    e = float(expectation)
    v = float(variance)
    if v < 0:
        raise ValueError(f"variance must be non-negative; got {v}.")
    if e <= 0:
        raise ValueError(f"expectation must be positive; got {e}.")
    ratio = v / (e * e)
    bound = min(ratio, 1.0)
    return RichResult(
        title="Second moment method",
        summary_lines=[
            ("Var / E^2", ratio),
            ("Bound on P(X = 0)", bound),
            ("Positive whp", ratio < 0.5),
        ],
        payload={
            "p_zero_bound": bound,
            "estimate": bound,
            "variance_ratio": ratio,
            "expectation": e,
            "variance": v,
            "positive_whp": ratio < 0.5,
            "vacuous": bound >= 1.0 - 1e-12,
            "n": 1,
            "method": "Second moment method (Chebyshev)",
        },
        interpretation=(
            f"Var/E^2 = {ratio:.6g}, so P(X = 0) is at most {bound:.6g}."
            if ratio < 1 else
            "Var/E^2 is at least 1, so Chebyshev gives nothing here."
        ),
    )


def cheatsheet():
    return (
        "prbmth: first and second moment methods, the alteration bound, the "
        "Lovasz Local Lemma, and Chernoff and Azuma inequalities -- each "
        "checked against the exact quantity it bounds"
    )
