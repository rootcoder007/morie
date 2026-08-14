# morie.fn -- function file (rootcoder007/morie)
r"""Higher-order targeted loss-based estimation.

A TMLE's asymptotics rest on a **first-order** representation: the
estimator equals its target plus an empirical mean of the efficient
influence curve plus a second-order remainder, and the argument works
only if that remainder is :math:`o_P(n^{-1/2})`. That in turn demands
nuisance estimators converging fast enough -- often faster than
:math:`n^{-1/4}` when a density enters the parameter, which in high
dimensions or under weak smoothness is simply not available.

**The generalisation.** Carry the expansion further. Where the
first-order representation uses one gradient, the higher-order version
adds a **second-order term**: a kernel :math:`D_2(P)(o_1, o_2)`
integrated against the product measure, capturing the curvature the
first-order term misses. Targeting is then done against both, and the
remainder that must vanish is now *third* order -- so the same
conclusion follows under weaker conditions on the nuisance estimators.

**Two practical consequences the chapter names.** Estimators built this
way behave better in finite samples, because the neglected curvature is
exactly what a first-order estimator mistakes for signal when the
sample is small; and they are asymptotically efficient under less
restrictive conditions, because the rate requirement is relaxed.

**What it costs.** The second-order term is a :math:`U`-statistic over
pairs, so the work is :math:`O(n^2)` rather than :math:`O(n)`, and it
requires a second kernel that must itself be estimated -- which is why
Chap. 8's numerical machinery matters here. ``remainder_order``
reports which order of remainder is being relied on, since that is the
assumption doing the work.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 26 (Carone,
Diaz & van der Laan): generalising the TMLE framework to use
higher-order rather than first-order asymptotic representations, to
provide guidelines for estimators with sound finite-sample behaviour
that are asymptotically efficient under less restrictive conditions;
the requirement that the second-order remainder tend to zero faster
than n^{-1/2} for the first-order representation to be useful; and the
example that when the density of the data-generating distribution is
directly involved in the target parameter, a density estimator
converging faster than n^{-1/4} in a suitable norm is often required
to make that remainder negligible.

Carone, M., Diaz, I. & van der Laan, M. J. (2018) "Higher-Order
Targeted Loss-Based Estimation", in *Targeted Learning in Data
Science*, Springer, 483-510, doi:10.1007/978-3-319-65304-4_26.

Robins, J., Li, L., Tchetgen Tchetgen, E. & van der Vaart, A. (2008)
"Higher order influence functions and minimax estimation of nonlinear
functionals", in *Probability and Statistics: Essays in Honor of
David A. Freedman*, IMS, 335-421, doi:10.1214/193940307000000527.
Higher-order influence functions.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["first_order_expansion", "second_order_term",
           "higher_order_estimate", "remainder_order",
           "rate_requirement"]

_EPS = 1e-12


def first_order_expansion(D1, psi_plugin):
    r""":math:`\psi_n = \Psi(P_n) + P_n D_1 + R_2`."""
    d = [float(v) for v in k.vec(D1)]
    n = len(d)
    if n < 2:
        raise ValueError("tlhoest: at least 2 observations are needed")
    m = sum(d) / n
    return {"estimate": float(psi_plugin) + m, "mean_D1": m,
            "order": 1,
            "note": "valid only if the SECOND-order remainder is "
                    "o(n^{-1/2})"}


def second_order_term(D2_kernel, O, exclude_diagonal=True):
    r"""The U-statistic :math:`\frac{1}{n(n-1)}\sum_{i \ne j}
    D_2(o_i, o_j)`.

    The diagonal is excluded because :math:`D_2(o,o)` is not an
    independent pair and including it introduces exactly the bias the
    term is meant to remove.
    """
    obs = list(O)
    n = len(obs)
    if n < 2:
        raise ValueError("tlhoest: a second-order term needs at least "
                         "2 observations")
    tot, m = 0.0, 0
    for i in range(n):
        for j in range(n):
            if exclude_diagonal and i == j:
                continue
            tot += float(D2_kernel(obs[i], obs[j]))
            m += 1
    return {"value": tot / m, "n_pairs": m, "cost": "O(n^2)",
            "note": "a U-statistic over PAIRS, which is where the "
                    "curvature the first-order term misses lives"}


def higher_order_estimate(psi_plugin, D1, D2_kernel, O):
    r"""First-order plus second-order correction.

    The remainder that must now vanish is THIRD order, which is why
    the nuisance rate requirement relaxes.
    """
    fo = first_order_expansion(D1, psi_plugin)
    so = second_order_term(D2_kernel, O)
    return RichResult(payload={
        "estimate": fo["estimate"] + so["value"],
        "psi": fo["estimate"] + so["value"],
        "first_order": fo["estimate"],
        "second_order_correction": so["value"],
        "n_pairs": so["n_pairs"],
        "method": "higher-order targeted loss-based estimation; van "
                  "der Laan & Rose (2018) Chap. 26",
        "note": "the remainder that must be o(n^{-1/2}) is now THIRD "
                "order",
    })


def rate_requirement(order, n=1000):
    r"""The nuisance rate a given expansion order demands.

    First order needs the product of nuisance errors below
    :math:`n^{-1/2}` -- roughly :math:`n^{-1/4}` each. Second order
    needs the *triple* product, so about :math:`n^{-1/6}` each: a much
    weaker demand, which is the whole gain.
    """
    o = int(order)
    if o < 1:
        raise ValueError("tlhoest: the order must be at least 1")
    per = 0.5 / (o + 1)
    return {"order": o, "required_rate_per_nuisance": per,
            "example_n": int(n),
            "error_at_that_rate": int(n) ** (-per),
            "note": "each additional order relaxes the per-nuisance "
                    "rate requirement"}


def remainder_order(order):
    r"""Which remainder the argument is relying on."""
    o = int(order)
    return {"expansion_order": o, "remainder_order": o + 1,
            "must_be": "o(n^{-1/2})",
            "note": "naming the order is naming the assumption doing "
                    "the work"}


def cheatsheet():
    return ("tlhoest: TMLE's first-order representation works only if "
            "the SECOND-order remainder is o(n^{-1/2}), which forces "
            "nuisance rates faster than n^{-1/4} -- unavailable in "
            "high dimensions or under weak smoothness. Carry the "
            "expansion further: add a second-order kernel D_2 "
            "integrated over PAIRS (a U-statistic, O(n^2), diagonal "
            "excluded), target against both, and now only the THIRD- "
            "order remainder must vanish. Better finite-sample "
            "behaviour, efficiency under weaker conditions, at "
            "quadratic cost and needing a second kernel estimated.")


# compact alias per ledger/NAMING.md
higherordertmle = higher_order_estimate
