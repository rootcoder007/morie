# morie.fn -- function file (rootcoder007/morie)
r"""Augmented outcome-weighted learning for dynamic treatment regimens.

Outcome-weighted learning turns "find the treatment rule with the best
expected outcome" into a weighted classification problem: label each
subject by the treatment they actually received, weight them by their
outcome over the randomisation probability, and the classifier that
minimises weighted error is the rule that maximises value. It works,
and it has two defects that AOL removes.

**Defect one: negative outcomes.** The weights are
:math:`R_i / \pi(A_i \mid H_i)`, and a weighted classifier needs
non-negative weights. OWL's answer is to add a large constant to every
outcome. That is not free -- the constant changes the relative weights,
and a big one flattens them all toward equality, which is numerically
unstable and throws away the signal. AOL instead weights by a
**residual**,

.. math:: W_i = \frac{R_i - m(H_i)}{\pi(A_i \mid H_i)},

where :math:`m` is a fitted prognostic model. A negative residual is
not a problem: it flips the label. A subject who did worse than
predicted under the treatment they got is evidence *for the other
arm*, and encoding that as a sign is more honest than shifting it away.

**Defect two: discarded subjects in multiple stages.** In a K-stage
regimen, OWL's backward induction can only use subjects whose *later*
treatments were already optimal, because the weight at stage :math:`k`
must be the optimal outcome increment from the future. Everyone else is
thrown away, and the proportion discarded compounds across stages. AOL
augments the future value with a model-based prediction for the
subjects whose later treatments were not optimal, so **all** subjects
contribute at every stage.

**What the augmentation costs and what it does not.** It does not cost
correctness: AOL is proved to give the right optimal regimen even when
the augmentation regression is misspecified, so the robustness of the
nonparametric approach survives. It buys variance: removing the
prognostic part of the outcome shrinks the spread of the weights, and
the paper's theory gives the same asymptotic bias as OWL with strictly
smaller stochastic error.

**Why residualising reduces variance, concretely.** Suppose the outcome
is mostly prognosis -- a large :math:`m(H)` common to both arms -- plus
a small treatment effect. OWL's weights are dominated by prognosis,
which carries no information about *which* treatment is better; it is
noise in the objective. Subtracting :math:`m` leaves the part that does
discriminate. The anchor measures this directly rather than asserting
it: with a strong prognostic signal the OWL weights have a far larger
coefficient of variation than the AOL weights, and the AOL rule is the
one that recovers the planted decision boundary.

References
----------
Liu, Y., Wang, Y., Kosorok, M. R., Zhao, Y. & Zeng, D. (2018)
"Augmented outcome-weighted learning for estimating optimal dynamic
treatment regimens", *Statistics in Medicine*,
doi:10.1002/sim.7844. Sec. 1 (the four stated contributions:
negative outcomes without an additive constant, residual weights,
combining nonparametric robustness with model-based augmentation, and
the same asymptotic bias as OWL with smaller stochastic variability),
Sec. 2.1 (K-stage notation, the value function, and the OWL
formulation) and Sec. 2 (AOL for single and multiple stages). Volume
and pages are not printed in the file held locally.

Zhao, Y., Zeng, D., Rush, A. J. & Kosorok, M. R. (2012) "Estimating
individualized treatment rules using outcome weighted learning",
*Journal of the American Statistical Association* 107(499),
1106-1118, doi:10.1080/01621459.2012.695674. The single-stage OWL
this augments.

Zhao, Y.-Q., Zeng, D., Laber, E. B. & Kosorok, M. R. (2015) "New
statistical learning methods for estimating optimal dynamic treatment
regimes", *Journal of the American Statistical Association* 110(510),
583-598, doi:10.1080/01621459.2014.937488. The multi-stage
backward-induction OWL whose discarding of subjects AOL removes.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["owl_weights", "aol_weights", "weighted_rule",
           "fit_aol", "fit_stages", "regimen_value"]

_EPS = 1e-12
_METHODS = ("aol", "owl")


def _check(R, A, H, propensity):
    r = [float(v) for v in k.vec(R)]
    a = [int(v) for v in k.vec(A)]
    Hm = [[float(v) for v in row] for row in k.mat(H)]
    n = len(r)
    if not (len(a) == len(Hm) == n):
        raise ValueError("awltrn: R, A and H must agree in length "
                         "(%d, %d, %d)" % (n, len(a), len(Hm)))
    if n < 4:
        raise ValueError("awltrn: need at least 4 subjects, got %d"
                         % n)
    for v in a:
        if v not in (-1, 1):
            raise ValueError("awltrn: treatments must be coded -1/+1, "
                             "got %r" % (v,))
    p = ([0.5] * n if propensity is None
         else ([float(propensity)] * n
               if isinstance(propensity, (int, float))
               else [float(v) for v in k.vec(propensity)]))
    if len(p) != n:
        raise ValueError("awltrn: %d propensities for %d subjects"
                         % (len(p), n))
    if any(not 0.0 < v < 1.0 for v in p):
        raise ValueError("awltrn: randomisation probabilities must "
                         "lie strictly in (0, 1)")
    return r, a, Hm, p, n


def owl_weights(R, A, H, propensity=None, shift=None):
    r"""OWL weights :math:`R_i/\pi_i`, with the additive constant.

    ``shift`` is the constant added to make every outcome
    non-negative. Left as ``None`` it is chosen as the smallest value
    that does so -- and the point of AOL is that this choice is
    consequential, not a formality, so the value used is reported.
    """
    r, a, Hm, p, n = _check(R, A, H, propensity)
    c = (0.0 if min(r) >= 0.0 else -min(r)) if shift is None \
        else float(shift)
    w = [(r[i] + c) / p[i] for i in range(n)]
    if any(v < 0.0 for v in w):
        raise ValueError("awltrn: OWL weights must be non-negative; "
                         "increase shift (smallest weight %.4g)"
                         % min(w))
    lab = list(a)
    m = sum(w) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in w) / max(n - 1, 1))
    return {"weights": w, "labels": lab, "shift": c,
            "cv": sd / m if m > _EPS else float("inf"),
            "note": "the additive constant changes the RELATIVE "
                    "weights; a large one flattens them toward "
                    "equality and discards signal"}


def aol_weights(R, A, H, propensity=None, prognostic=None,
                ridge=1e-8):
    r"""AOL weights :math:`(R_i - m(H_i))/\pi_i`.

    ``prognostic`` is :math:`m(H)`; if omitted it is fitted by least
    squares on :math:`H` alone -- deliberately **without** treatment,
    since the point is to remove the part of the outcome common to
    both arms. A negative residual flips the label rather than being
    shifted away.
    """
    r, a, Hm, p, n = _check(R, A, H, propensity)
    if prognostic is None:
        D = k.design(Hm, n)
        beta = k.lstsq(D, r, ridge)
        m = [sum(D[i][j] * beta[j] for j in range(len(beta)))
             for i in range(n)]
    else:
        m = [float(v) for v in k.vec(prognostic)]
        if len(m) != n:
            raise ValueError("awltrn: %d prognostic values for %d "
                             "subjects" % (len(m), n))
    resid = [r[i] - m[i] for i in range(n)]
    w = [abs(resid[i]) / p[i] for i in range(n)]
    lab = [a[i] if resid[i] >= 0.0 else -a[i] for i in range(n)]
    mw = sum(w) / n
    sd = math.sqrt(sum((v - mw) ** 2 for v in w) / max(n - 1, 1))
    return {"weights": w, "labels": lab, "residual": resid,
            "prognostic": m, "n_flipped": sum(1 for i in range(n)
                                              if resid[i] < 0.0),
            "cv": sd / mw if mw > _EPS else float("inf"),
            "note": "a negative residual FLIPS the label -- doing "
                    "worse than predicted under the arm received is "
                    "evidence for the other arm"}


def weighted_rule(H, labels, weights, ridge=1e-6):
    r"""A weighted linear classifier, returned as a decision function.

    Minimises weighted squared error against the labels, which is the
    surrogate used here for the weighted 0-1 loss. The sign of the fit
    is the rule.
    """
    Hm = [[float(v) for v in row] for row in k.mat(H)]
    n = len(Hm)
    if not (len(labels) == len(weights) == n):
        raise ValueError("awltrn: H, labels and weights must agree in "
                         "length")
    if any(float(v) < 0.0 for v in weights):
        raise ValueError("awltrn: weights must be non-negative")
    D = k.design(Hm, n)
    fit = k.wls(Hm, [float(v) for v in labels],
                [float(v) for v in weights], ridge=ridge)
    b = fit["coef"]

    def rule(x):
        s = b[0] + sum(float(x[j]) * b[j + 1]
                       for j in range(len(b) - 1))
        return 1 if s >= 0.0 else -1

    return {"rule": rule, "coef": b}


def regimen_value(R, A, H, rule, propensity=None):
    r"""Inverse-probability value of a single-stage rule."""
    r, a, Hm, p, n = _check(R, A, H, propensity)
    num = sum(r[i] * (1.0 if rule(Hm[i]) == a[i] else 0.0) / p[i]
              for i in range(n))
    den = sum((1.0 if rule(Hm[i]) == a[i] else 0.0) / p[i]
              for i in range(n))
    if den <= _EPS:
        raise ValueError("awltrn: no subject's observed treatment "
                         "agrees with the rule")
    return num / den


def fit_aol(R, A, H, propensity=None, method="aol", prognostic=None,
            shift=None, ridge=1e-6):
    r"""Single-stage AOL (or plain OWL, for comparison)."""
    if method not in _METHODS:
        raise ValueError("awltrn: method must be aol or owl, got %r"
                         % (method,))
    if method == "aol":
        w = aol_weights(R, A, H, propensity=propensity,
                        prognostic=prognostic)
    else:
        w = owl_weights(R, A, H, propensity=propensity, shift=shift)
    cl = weighted_rule(H, w["labels"], w["weights"], ridge=ridge)
    v = regimen_value(R, A, H, cl["rule"], propensity=propensity)
    return RichResult(payload={
        "estimate": v, "value": v, "rule": cl["rule"],
        "coef": cl["coef"], "weights": w["weights"],
        "labels": w["labels"], "weight_cv": w["cv"],
        "method": method, "n": len(w["weights"]),
        "n_flipped": w.get("n_flipped"),
        "shift": w.get("shift"),
        "reference": "Liu, Wang, Kosorok, Zhao & Zeng (2018)",
    })


def fit_stages(stages, propensity=None, ridge=1e-6):
    r"""Backward induction across K stages, using **every** subject.

    ``stages`` is a list of ``(R_k, A_k, H_k)`` in order. Working
    backwards, the pseudo-outcome carried into stage :math:`k` is the
    reward at :math:`k` plus the value achieved downstream -- for
    subjects whose later treatment matched the estimated optimum that
    is their observed future reward, and for the rest it is the
    model-based prediction. That augmentation is what stops the
    sample from shrinking stage by stage.
    """
    if not stages:
        raise ValueError("awltrn: no stages given")
    K = len(stages)
    n = len(k.vec(stages[0][0]))
    for j, (Rk, Ak, Hk) in enumerate(stages):
        if len(k.vec(Rk)) != n or len(k.vec(Ak)) != n \
                or len(k.mat(Hk)) != n:
            raise ValueError("awltrn: stage %d has a different number "
                             "of subjects" % j)
    future = [0.0] * n
    rules, used = [], []
    for j in range(K - 1, -1, -1):
        Rk, Ak, Hk = stages[j]
        rv = [float(v) for v in k.vec(Rk)]
        pseudo = [rv[i] + future[i] for i in range(n)]
        fit = fit_aol(pseudo, Ak, Hk, propensity=propensity,
                      method="aol", ridge=ridge)
        rules.insert(0, fit["rule"])
        used.insert(0, n)          # every subject contributes
        Hm = k.mat(Hk)
        av = [int(v) for v in k.vec(Ak)]
        # augmentation: observed future for concordant subjects, the
        # fitted prognostic prediction for the rest
        aw = aol_weights(pseudo, Ak, Hk, propensity=propensity)
        future = [pseudo[i] if fit["rule"](Hm[i]) == av[i]
                  else aw["prognostic"][i] for i in range(n)]
    return RichResult(payload={
        "estimate": sum(future) / n, "rules": rules,
        "n_stages": K, "n_used_per_stage": used, "n": n,
        "method": "augmented backward induction; Liu et al. (2018) "
                  "Sec. 2",
        "note": "every subject contributes at every stage -- OWL's "
                "backward induction keeps only those whose later "
                "treatments were optimal",
    })


def cheatsheet():
    return ("awltrn: AOL. OWL weights R/pi and needs R >= 0, so it "
            "ADDS A CONSTANT -- which changes the relative weights "
            "and flattens them. AOL weights |R - m(H)|/pi and lets a "
            "negative residual FLIP THE LABEL instead. Removing the "
            "prognostic part cuts weight variance without changing "
            "the asymptotic bias, and stays correct even if m is "
            "misspecified. Multi-stage: augmentation keeps ALL "
            "subjects at every stage rather than discarding those "
            "whose later treatments were not optimal.")


# compact alias per ledger/NAMING.md
augmentedowl = fit_aol

# public names resolved by fn/_lazy_map.json
augmented_owl = fit_aol
