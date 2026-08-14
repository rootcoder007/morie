# morie.fn -- function file (rootcoder007/morie)
r"""HAL-TMLE: efficiency under weak conditions.

A TMLE is asymptotically efficient when two things hold: the nuisance
estimators converge fast enough that the second-order remainder is
:math:`o_P(n^{-1/2})`, and the estimated functions stay in a
Donsker class so the empirical process term is controlled. Both are
usually *assumed*. The point of this chapter is that using HAL as the
initial estimator -- or a super learner whose library contains HAL --
makes both hold rather than assuming them.

**Why the remainder is controlled.** The second-order remainder of a
doubly robust parameter is a product of the two nuisance errors. If
each converges faster than :math:`n^{-1/4}`, the product is
:math:`o_P(n^{-1/2})` and vanishes at the rate required. HAL clears
:math:`n^{-1/4}` under only cadlag-and-finite-variation, so the
condition is met without a smoothness assumption and without a
correctly specified model.

**Why the empirical process term is controlled.** HAL fits live in the
class of cadlag functions with variation norm bounded by a universal
constant, and that class is Donsker. So the usual Donsker condition is
satisfied by construction rather than posited -- and where one prefers
not to rely on it at all, cross-validated TMLE removes the requirement
entirely.

**What is still required, and it is not weak.** Strong positivity. The
remainder bound involves the inverse treatment probability, and no
rate on the nuisance estimators rescues an estimand the data cannot
identify. ``remainder_bound`` makes that dependence explicit:
:math:`R_2 \le \|\bar Q - \bar Q_0\|\cdot\|g - g_0\|/\delta`, so as
:math:`\delta \to 0` the bound diverges however good the fits are.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 7 (a TMLE
that uses the HAL estimator as initial estimator, or a super learner
whose library contains it, is asymptotically efficient under very weak
regularity conditions, as long as the strong positivity assumption
holds). Chap. 6 (the HAL rate faster than n^{-1/4}). Chap. 4 (the
second-order remainder of the longitudinal parameter and the role of
the positivity bound in controlling it; and the note that the Donsker
class assumption can be avoided by using cross-validated TMLE).

van der Laan, M. J. (2017) "A generally efficient targeted minimum
loss based estimator based on the highly adaptive lasso",
*International Journal of Biostatistics* 13(2), 20150097,
doi:10.1515/ijb-2015-0097.

Zheng, W. & van der Laan, M. J. (2011) "Cross-Validated Targeted
Minimum-Loss-Based Estimation", in *Targeted Learning*, Springer,
459-474, doi:10.1007/978-1-4419-9782-1_27. CV-TMLE, which removes the
Donsker requirement.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["remainder_bound", "rate_condition", "efficiency_check",
           "cv_tmle_split"]

_EPS = 1e-12


def rate_condition(rate_Q, rate_g, n):
    r"""Is the product of the two nuisance errors
    :math:`o_P(n^{-1/2})`?

    Rates are given as exponents: ``rate_Q = 0.25`` means
    :math:`n^{-1/4}`. The product condition is
    :math:`r_Q + r_g > 1/2`, which two estimators at exactly
    :math:`n^{-1/4}` only just fail to satisfy -- hence "faster than
    :math:`n^{-1/4}`", not "at".
    """
    a, b = float(rate_Q), float(rate_g)
    if a <= 0 or b <= 0:
        raise ValueError("tlhaltm: rates must be positive exponents")
    return {"sum": a + b, "required": 0.5, "satisfied": a + b > 0.5,
            "product_order": (int(n)) ** (-(a + b)),
            "root_n_order": (int(n)) ** (-0.5),
            "note": "two estimators at exactly n^{-1/4} sit ON the "
                    "boundary; HAL's n^{-1/3} clears it"}


def remainder_bound(err_Q, err_g, delta):
    r"""The second-order remainder bound.

    :math:`|R_2| \le \|\bar Q - \bar Q_0\|\,\|g - g_0\|/\delta` --
    a product of errors divided by the positivity bound, so perfect
    fits do not save a violated positivity assumption.
    """
    d = float(delta)
    if not 0.0 < d <= 1.0:
        raise ValueError("tlhaltm: the positivity bound delta must "
                         "lie in (0,1], got %r" % (delta,))
    return {"bound": float(err_Q) * float(err_g) / d,
            "delta": d, "err_Q": float(err_Q), "err_g": float(err_g),
            "note": "the bound diverges as delta -> 0 however small "
                    "the nuisance errors are"}


def efficiency_check(err_Q, err_g, delta, n, donsker=True):
    r"""Both conditions together, reported separately.

    Keeping them separate matters: they fail for different reasons and
    are fixed by different means -- rate by a better learner,
    empirical process by CV-TMLE.
    """
    r = remainder_bound(err_Q, err_g, delta)
    root_n = 1.0 / math.sqrt(int(n))
    return RichResult(payload={
        "estimate": r["bound"],
        "remainder_bound": r["bound"], "root_n": root_n,
        "remainder_negligible": r["bound"] < root_n,
        "donsker_satisfied": bool(donsker),
        "efficient": r["bound"] < root_n and bool(donsker),
        "positivity_delta": float(delta),
        "method": "HAL-TMLE efficiency conditions; van der Laan & "
                  "Rose (2018) Chap. 7",
        "note": "HAL supplies BOTH -- the rate, and a bounded "
                "variation-norm class that is Donsker; CV-TMLE "
                "removes the second requirement entirely",
    })


def cv_tmle_split(n, V=10, seed=0):
    r"""The sample splits CV-TMLE uses to drop the Donsker condition.

    The initial fit is trained on the training split and evaluated on
    the validation split, so the empirical process term involves a
    function fixed independently of the data it is averaged over.
    """
    if int(V) < 2 or int(V) > int(n):
        raise ValueError("tlhaltm: V must lie in 2..%d, got %d"
                         % (n, V))
    rng = np.random.default_rng(seed)
    idx = list(range(int(n)))
    for i in range(len(idx) - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    folds = [sorted(idx[v::int(V)]) for v in range(int(V))]
    return {"folds": folds,
            "training": [sorted(set(range(int(n))) - set(f))
                         for f in folds],
            "V": int(V),
            "note": "the fit is independent of the validation sample, "
                    "so no Donsker condition is needed"}


def cheatsheet():
    return ("tlhaltm: TMLE is efficient when (a) the second-order "
            "remainder -- a PRODUCT of the two nuisance errors -- is "
            "o(n^-1/2), and (b) the fits stay in a Donsker class. HAL "
            "supplies both: its rate beats n^{-1/4}, so the product "
            "clears n^{-1/2}, and bounded-variation cadlag functions "
            "form a Donsker class. Neither is assumed. What IS still "
            "required is STRONG POSITIVITY: the remainder bound "
            "carries a 1/delta, so it diverges as delta -> 0 no "
            "matter how good the fits are. CV-TMLE drops (b) "
            "outright.")


# compact alias per ledger/NAMING.md
haltmle = efficiency_check
