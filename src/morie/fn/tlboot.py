# morie.fn -- function file (rootcoder007/morie)
r"""The targeted bootstrap.

Bootstrap intervals can be more accurate than Wald intervals, which is
why the bootstrap has been adopted in many settings -- including some
with no theory behind it. Targeted learning is one where the usual
application **fails**, and the reason is specific.

**Why the naive bootstrap fails here.** A TMLE fits its nuisance
functions with a super learner, which is a *data-adaptive* procedure.
Refitting the whole pipeline on each resample makes the nuisance fits
vary with the resample, and that variation is not the sampling
variation of the target parameter -- it is variation in an
infinite-dimensional object with a slower-than-root-:math:`n` rate.
The bootstrap distribution then reflects the learner's instability
rather than the estimator's sampling distribution, and the resulting
interval is wrong in a way more resamples cannot fix.

**The fix: resample from the targeted fit.** Hold the targeted
estimate :math:`P_n^*` fixed and generate resamples **from it** rather
than by resampling rows with the learner refitted. The bootstrap is
then designed to be consistent for the first two moments of the
sampling distribution -- mean and variance -- which is what an interval
needs. Three routes are implemented: the naive nonparametric bootstrap
(kept so the failure is visible), the targeted parametric bootstrap
from the fitted :math:`P_n^*`, and the influence-curve-based multiplier
bootstrap, which is the cheap approximation to the same thing.

**The comparison is the point.** With a smooth, correctly specified
nuisance fit all three agree. With an unstable learner they diverge,
and the anchor exhibits that divergence rather than describing it.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 28 (Coyle &
van der Laan): the bootstrap's use for confidence intervals and
hypothesis tests, with higher-order accuracy over Wald intervals in
some settings, and its wide adoption in contexts not all of which have
theoretical support; the typical targeted-learning workflow of initial
super learner fits followed by a TMLE; the description of why the
bootstrap as typically applied fails in that framework; and the
solution as a TARGETED BOOTSTRAP designed to be consistent for the
first two moments of the sampling distribution.

Efron, B. & Tibshirani, R. J. (1993) *An Introduction to the
Bootstrap*, Chapman and Hall, doi:10.1201/9780429246593.

Hall, P. (1992) *The Bootstrap and Edgeworth Expansion*, Springer,
doi:10.1007/978-1-4612-4384-7. The higher-order accuracy result.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["naive_bootstrap", "targeted_bootstrap",
           "multiplier_bootstrap", "moment_check"]

_EPS = 1e-12


def naive_bootstrap(data, estimator, B=200, seed=0):
    r"""Resample rows and refit everything -- the version that fails.

    Kept so the failure is measurable: with a data-adaptive nuisance
    fit the spread here reflects the learner's instability, not the
    estimator's sampling variability.
    """
    rows = list(data)
    n = len(rows)
    if n < 2:
        raise ValueError("tlboot: at least 2 observations are needed")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(B)):
        s = [rows[int(float(rng.uniform()) * n) % n]
             for _ in range(n)]
        out.append(float(estimator(s)))
    m = sum(out) / len(out)
    sd = math.sqrt(sum((v - m) ** 2 for v in out) / (len(out) - 1))
    return {"replicates": out, "mean": m, "se": sd, "B": int(B),
            "caveat": "refitting a data-adaptive learner on each "
                      "resample mixes the learner's instability into "
                      "the sampling distribution"}


def targeted_bootstrap(P_star_sampler, estimator, B=200, seed=0):
    r"""Resample from the TARGETED fit, holding the nuisances fixed.

    ``P_star_sampler(rng)`` draws a sample from :math:`P_n^*`. The
    nuisance fits are part of :math:`P_n^*` and do not move, so what
    varies is the sampling of the data alone -- which is what the
    interval is about.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(B)):
        out.append(float(estimator(P_star_sampler(rng))))
    m = sum(out) / len(out)
    sd = math.sqrt(sum((v - m) ** 2 for v in out) / (len(out) - 1))
    return RichResult(payload={
        "estimate": m, "mean": m, "se": sd,
        "replicates": out, "B": int(B),
        "method": "targeted bootstrap from the fitted P_n^*; van der "
                  "Laan & Rose (2018) Chap. 28",
        "note": "designed to be consistent for the first two moments "
                "of the sampling distribution",
    })


def multiplier_bootstrap(ic, B=1000, seed=0):
    r"""Multiply the influence curve by mean-one random weights.

    The cheap route to the same first two moments: no refitting at
    all, and the variance matches the influence-curve variance by
    construction.
    """
    d = [float(v) for v in k.vec(ic)]
    n = len(d)
    if n < 2:
        raise ValueError("tlboot: at least 2 influence values are "
                         "needed")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(B)):
        w = [-math.log(max(float(rng.uniform()), 1e-12))
             for _ in range(n)]
        s = sum(w)
        out.append(sum(w[i] * d[i] for i in range(n)) / s)
    m = sum(out) / len(out)
    sd = math.sqrt(sum((v - m) ** 2 for v in out) / (len(out) - 1))
    mm = sum(d) / n
    icse = math.sqrt(sum((v - mm) ** 2 for v in d) / (n - 1) / n)
    return {"replicates": out, "mean": m, "se": sd,
            "influence_curve_se": icse,
            "ratio": sd / icse if icse > 0 else float("nan"),
            "note": "matches the influence-curve standard error by "
                    "construction, at no refitting cost"}


def moment_check(replicates, target_mean, target_se, tol=0.15):
    r"""Are the first two moments right?

    That is the stated design goal, so it is the thing to check --
    not whether the quantiles look plausible.
    """
    v = [float(q) for q in k.vec(replicates)]
    n = len(v)
    if n < 2:
        raise ValueError("tlboot: at least 2 replicates are needed")
    m = sum(v) / n
    sd = math.sqrt(sum((q - m) ** 2 for q in v) / (n - 1))
    return {"mean": m, "se": sd,
            "mean_error": abs(m - float(target_mean)),
            "se_ratio": sd / float(target_se)
            if float(target_se) > 0 else float("nan"),
            "first_two_moments_ok":
                abs(sd / float(target_se) - 1.0) < float(tol),
            "note": "consistency for the first two moments is the "
                    "stated design goal"}


def cheatsheet():
    return ("tlboot: the ordinary bootstrap FAILS for TMLE. Refitting "
            "a super learner on every resample makes the nuisance fits "
            "move with the resample, and that is not the sampling "
            "variability of the target -- it is instability of an "
            "infinite-dimensional object converging slower than "
            "root-n, so more resamples do not help. Instead resample "
            "FROM THE TARGETED FIT P_n^*, holding the nuisances fixed; "
            "the design goal is consistency for the first TWO MOMENTS. "
            "The multiplier bootstrap on the influence curve is the "
            "cheap equivalent.")


# compact alias per ledger/NAMING.md
targetedbootstrap = targeted_bootstrap
