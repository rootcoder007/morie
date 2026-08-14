# morie.fn -- function file (rootcoder007/morie)
r"""TMLE for a bounded continuous or count outcome.

TMLE for a binary outcome uses the Bernoulli log-likelihood loss and a
logistic submodel, which keeps the updated fit inside :math:`[0,1]`.
For a count or bounded continuous outcome the obvious move -- a linear
fluctuation with squared-error loss -- is available and worse, for a
reason worth being precise about: the linear update is not bounded, so
the targeted fit can leave the parameter space entirely, and the
resulting substitution estimator can fall outside the range the outcome
can take.

**The fix is a change of scale, not of method.** Map the outcome to the
unit interval,

.. math:: Y^* = \frac{Y - a}{b - a},

with :math:`[a,b]` a known bound (for a count with an offset, the
observed range serves). Run the *binary* TMLE machinery on
:math:`Y^*` -- log-likelihood loss, logistic submodel, clever covariate
-- and map the estimate back. The quasi-log-likelihood loss is a valid
loss for a continuous outcome in :math:`[0,1]` even though :math:`Y^*`
is not Bernoulli, which is what makes the borrowing legitimate rather
than a heuristic.

**Two properties follow, and the anchor checks both.** The targeted fit
stays inside the bounds by construction; and the estimator is still
doubly robust and solves the efficient influence curve equation on the
original scale, because the rescaling is affine and the influence curve
transforms with it.

**For counts specifically**, an exposure offset :math:`\log(t_i)`
belongs in the outcome regression, and the estimand is a rate. The
bound is then :math:`[0, \max(Y/t)]` and the same argument applies.

References
----------
Gruber, S. & van der Laan, M. J. (2010) "A Targeted Maximum Likelihood
Estimator of a Causal Effect on a Bounded Continuous Outcome",
*The International Journal of Biostatistics* 6(1), Article 26,
doi:10.2202/1557-4679.1260. The rescaling to the unit interval, the
quasi-log-likelihood loss for a continuous outcome in [0,1], the
logistic fluctuation, and the resulting respect for the parameter
space.

van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 4: the
Bernoulli log-likelihood loss and logistic submodel with the clever
covariate, and the general TMLE roadmap this specialises.

Note on provenance: the ledger previously cited this module to Lendle,
Schwab, Petersen & van der Laan (2017), "ltmle: An R Package
Implementing Targeted Minimum Loss-Based Estimation for Longitudinal
Data", *Journal of Statistical Software* 81(1), 1-21,
doi:10.18637/jss.v081.i01 -- a software paper that does not treat
count outcomes. That citation was wrong and has been replaced.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rescale", "unscale", "tmle_count_outcome",
           "linear_fluctuation_unsafe"]

_EPS = 1e-12


def _logit(p):
    q = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(q / (1.0 - q))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def rescale(y, lower=None, upper=None):
    r"""Map the outcome to :math:`[0,1]` by an affine transform."""
    v = [float(q) for q in k.vec(y)]
    if not v:
        raise ValueError("tmlcou: no outcomes given")
    a = float(lower) if lower is not None else min(v)
    b = float(upper) if upper is not None else max(v)
    if b <= a:
        raise ValueError("tmlcou: the upper bound must exceed the "
                         "lower one, got (%r, %r)" % (a, b))
    if any(q < a - _EPS or q > b + _EPS for q in v):
        raise ValueError("tmlcou: an outcome lies outside the stated "
                         "bounds")
    return {"scaled": [(q - a) / (b - a) for q in v],
            "lower": a, "upper": b, "range": b - a}


def unscale(value, lower, upper):
    r"""Invert the affine map -- the influence curve scales with it."""
    return float(value) * (float(upper) - float(lower)) + float(lower)


def linear_fluctuation_unsafe(Q, H, Y):
    r"""The linear update, kept so its failure is demonstrable.

    Squared-error loss with an additive fluctuation is unbounded: the
    updated fit can leave the outcome's range, and then so can the
    substitution estimator.
    """
    q = [float(v) for v in k.vec(Q)]
    h = [float(v) for v in k.vec(H)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    num = sum(h[i] * (y[i] - q[i]) for i in range(n))
    den = sum(h[i] * h[i] for i in range(n))
    e = num / den if den > _EPS else 0.0
    upd = [q[i] + e * h[i] for i in range(n)]
    return {"epsilon": e, "Q_star": upd,
            "out_of_range": sum(1 for v in upd
                                if v < 0.0 or v > 1.0),
            "caveat": "an additive fluctuation is not bounded"}


def tmle_count_outcome(y, D, X, offset=None, g=None, Q1=None,
                       Q0=None, lower=None, upper=None, iters=100):
    r"""TMLE of the mean-outcome contrast for a count or bounded
    outcome.

    ``offset`` supplies exposure time, in which case the estimand is a
    rate. Nuisance fits may be supplied; otherwise they are fitted by
    logistic and least-squares regression on ``X``.
    """
    yv = [float(v) for v in k.vec(y)]
    a = [float(v) for v in k.vec(D)]
    W = [[float(v) for v in r] for r in k.mat(X)]
    n = len(yv)
    if not (len(a) == len(W) == n):
        raise ValueError("tmlcou: the inputs differ in length")
    if any(v < 0.0 for v in yv):
        raise ValueError("tmlcou: a count outcome cannot be negative")
    if offset is not None:
        t = [float(v) for v in k.vec(offset)]
        if len(t) != n or any(v <= 0.0 for v in t):
            raise ValueError("tmlcou: the offset must be positive and "
                             "of the same length")
        yv = [yv[i] / t[i] for i in range(n)]
    sc = rescale(yv, lower, upper)
    ys = sc["scaled"]
    if g is None:
        des = k.design(W, n)
        b = k.logit_irls(des, a)
        gg = [min(max(_expit(sum(des[i][j] * b[j]
                                 for j in range(len(b)))),
                      0.01), 0.99) for i in range(n)]
    else:
        gg = [min(max(float(v), 1e-6), 1 - 1e-6) for v in k.vec(g)]
    if Q1 is None or Q0 is None:
        Xa = [[a[i]] + list(W[i]) for i in range(n)]
        co = k.wls(Xa, ys, [1.0] * n, 0.0)["coef"]

        def pred(av, i):
            row = [1.0, av] + list(W[i])
            return sum(row[j] * co[j] for j in range(len(co)))

        q1 = [min(max(pred(1.0, i), 1e-6), 1 - 1e-6)
              for i in range(n)]
        q0 = [min(max(pred(0.0, i), 1e-6), 1 - 1e-6)
              for i in range(n)]
    else:
        q1 = [min(max(float(v), 1e-6), 1 - 1e-6) for v in k.vec(Q1)]
        q0 = [min(max(float(v), 1e-6), 1 - 1e-6) for v in k.vec(Q0)]
    H = [a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
         for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]
    off = [_logit(v) for v in qa]
    e = 0.0
    for _ in range(int(iters)):
        p = [_expit(off[i] + e * H[i]) for i in range(n)]
        gr = sum(H[i] * (ys[i] - p[i]) for i in range(n))
        he = sum(H[i] * H[i] * p[i] * (1 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        step = gr / he
        e += step
        if abs(step) < 1e-12:
            break
    q1s = [_expit(_logit(q1[i]) + e / gg[i]) for i in range(n)]
    q0s = [_expit(_logit(q0[i]) - e / (1 - gg[i])) for i in range(n)]
    psi_s = sum(q1s[i] - q0s[i] for i in range(n)) / n
    psi = psi_s * sc["range"]
    d = []
    for i in range(n):
        qas = q1s[i] if a[i] == 1.0 else q0s[i]
        d.append((H[i] * (ys[i] - qas) + q1s[i] - q0s[i] - psi_s)
                 * sc["range"])
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "epsilon": e, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "solves_eic": abs(m) < 1e-6,
        "scale": (sc["lower"], sc["upper"]),
        "mean_treated": unscale(sum(q1s) / n, sc["lower"],
                                sc["upper"]),
        "mean_control": unscale(sum(q0s) / n, sc["lower"],
                                sc["upper"]),
        "in_range": all(0.0 <= v <= 1.0 for v in q1s + q0s),
        "rate_scale": offset is not None,
        "method": "TMLE on a bounded outcome by rescaling to [0,1] "
                  "with a logistic fluctuation; Gruber & van der Laan "
                  "(2010)",
        "note": "a LINEAR fluctuation would leave the parameter space; "
                "the logistic one cannot",
    })


def cheatsheet():
    return ("tmlcou: for a COUNT or bounded continuous outcome do not "
            "fluctuate linearly -- an additive update is unbounded and "
            "the targeted fit, and the estimate, can leave the range "
            "the outcome can take. Rescale Y to [0,1] by "
            "(Y - a)/(b - a), run the BINARY machinery "
            "(quasi-log-likelihood loss, logistic submodel, clever "
            "covariate), and map back. The quasi-loss is valid for a "
            "continuous outcome in [0,1] even though it is not "
            "Bernoulli. With an exposure offset the estimand is a "
            "RATE. Double robustness survives because the rescaling is "
            "affine.")


# compact alias per ledger/NAMING.md
tmlecountoutcome = tmle_count_outcome
