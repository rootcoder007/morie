# morie.fn -- function file (rootcoder007/morie)
r"""The sample average treatment effect.

In a cluster randomized trial the units are almost never a simple
random sample from a defined population: the target population is
hypothetical or ill-defined, and units are chosen for logistical
reasons. The **population** average treatment effect is then a
parameter of a superpopulation nobody sampled from -- neither well
defined nor easily interpretable.

The **sample** effect is the mean difference in counterfactual
outcomes for *the study units themselves*,

.. math:: \mathrm{SATE} = \frac{1}{n}\sum_{i=1}^{n}
          \big(Y_i(1) - Y_i(0)\big),

which is interpretable without inventing a superpopulation, and is
arguably the more relevant quantity when the units were not sampled
from one.

**It is not identifiable in finite samples**, and the chapter says so
plainly: the counterfactuals are not both observed for any unit. What
rescues it is that the TMLE for the *population* effect is consistent
and asymptotically linear for the sample effect too -- the same point
estimate serves both, and only the inference changes.

**The inference changes in one specific way.** The influence curve for
the population effect carries two pieces: the weighted residual term
and the term :math:`\bar Q_1 - \bar Q_0 - \psi`, which is the
variability of the *individual* effects across units. The sample
effect conditions on those units, so that second piece drops:

.. math:: IC^{S} \approx \Big(\frac{I(A=1)}{g}
          - \frac{I(A=0)}{1-g}\Big)(Y - \bar Q_A).

The resulting variance is therefore **smaller by the variance of the
conditional effect**, exactly, and the estimator is asymptotically
conservative for the sample effect. Where effect modification is
present -- where individual effects genuinely differ -- that gap is
large, and targeting the sample effect is where the precision and
power come from. The anchor computes both influence curves on the
same data and requires the difference to equal
:math:`\mathrm{var}(\bar Q_1 - \bar Q_0)`.

**Pair-matched trials** are handled by the same argument with the
matched-pair structure entering the variance estimate.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 12 (in
cluster randomized trials the units are not a simple random sample and
the target population is hypothetical or ill-defined, so the PATE may
be neither well defined nor easily interpretable; the SATE as the mean
difference in counterfactual outcomes for the study units; that the
SATE is not formally identifiable in finite samples but a TMLE for the
population effect is consistent and asymptotically linear for it with
an asymptotically conservative variance estimator; the conservative
influence curve dropping the Q1 - Q0 - psi term; the extension to
pair-matched trials; and the finding that with effect modification,
targeting the sample effect yields the most precision and power).

Balzer, L. B., Petersen, M. L. & van der Laan, M. J. (2016) "Targeted
estimation and inference for the sample average treatment effect in
trials with and without pair-matching", *Statistics in Medicine*
35(21), 3717-3732, doi:10.1002/sim.6965.

Imbens, G. W. (2004) "Nonparametric estimation of average treatment
effects under exogeneity: a review", *Review of Economics and
Statistics* 86(1), 4-29, doi:10.1162/003465304323023651. The
sample/population distinction.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sate_influence_curve", "pate_influence_curve",
           "variance_gap", "sate_tmle", "paired_variance"]

_EPS = 1e-12


def _check(A, Y, Q1, Q0, g):
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    gg = [float(v) for v in k.vec(g)]
    n = len(a)
    if not (len(y) == len(q1) == len(q0) == len(gg) == n):
        raise ValueError("tlsate: the inputs differ in length")
    if any(v <= 0.0 or v >= 1.0 for v in gg):
        raise ValueError("tlsate: the treatment probability must lie "
                         "strictly inside (0,1)")
    return a, y, q1, q0, gg, n


def pate_influence_curve(A, Y, Q1, Q0, g, psi):
    r"""The population influence curve: residual term PLUS
    :math:`\bar Q_1 - \bar Q_0 - \psi`."""
    a, y, q1, q0, gg, n = _check(A, Y, Q1, Q0, g)
    out = []
    for i in range(n):
        qa = q1[i] if a[i] == 1.0 else q0[i]
        h = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
        out.append(h * (y[i] - qa) + q1[i] - q0[i] - float(psi))
    return out


def sate_influence_curve(A, Y, Q1, Q0, g):
    r"""The sample influence curve: the residual term ONLY.

    Conditioning on the study units removes the between-unit
    variability of the individual effects.
    """
    a, y, q1, q0, gg, n = _check(A, Y, Q1, Q0, g)
    out = []
    for i in range(n):
        qa = q1[i] if a[i] == 1.0 else q0[i]
        h = a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
        out.append(h * (y[i] - qa))
    return out


def variance_gap(A, Y, Q1, Q0, g, psi):
    r"""The exact difference between the two variances.

    It is :math:`\mathrm{var}(\bar Q_1 - \bar Q_0)` -- the variability
    of the conditional effect -- which is why effect modification is
    where the precision gain lives.
    """
    a, y, q1, q0, gg, n = _check(A, Y, Q1, Q0, g)
    icp = pate_influence_curve(A, Y, Q1, Q0, g, psi)
    ics = sate_influence_curve(A, Y, Q1, Q0, g)

    def var(v):
        m = sum(v) / len(v)
        return sum((q - m) ** 2 for q in v) / (len(v) - 1)

    eff = [q1[i] - q0[i] for i in range(n)]
    return {"var_pate": var(icp), "var_sate": var(ics),
            "gap": var(icp) - var(ics),
            "var_conditional_effect": var(eff),
            "note": "the gap IS the variance of the conditional "
                    "effect; with no effect modification it is zero"}


def sate_tmle(A, Y, Q1, Q0, g):
    r"""The same point estimate, two inferences.

    The TMLE for the population effect is used unchanged; only the
    influence curve, and therefore the interval, differs.
    """
    a, y, q1, q0, gg, n = _check(A, Y, Q1, Q0, g)
    H = [a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
         for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]

    def logit(p):
        q = min(max(p, 1e-9), 1 - 1e-9)
        return math.log(q / (1 - q))

    def expit(x):
        return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0

    off = [logit(v) for v in qa]
    e = 0.0
    for _ in range(60):
        p = [expit(off[i] + e * H[i]) for i in range(n)]
        gr = sum(H[i] * (y[i] - p[i]) for i in range(n))
        he = sum(H[i] * H[i] * p[i] * (1 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        e += gr / he
    q1s = [expit(logit(q1[i]) + e / gg[i]) for i in range(n)]
    q0s = [expit(logit(q0[i]) - e / (1 - gg[i])) for i in range(n)]
    psi = sum(q1s[i] - q0s[i] for i in range(n)) / n
    icp = pate_influence_curve(a, y, q1s, q0s, gg, psi)
    ics = sate_influence_curve(a, y, q1s, q0s, gg)

    def se(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((q - m) ** 2 for q in v)
                         / (len(v) - 1) / len(v))

    sp, ss = se(icp), se(ics)
    return RichResult(payload={
        "estimate": psi, "psi": psi,
        "se_population": sp, "se_sample": ss,
        "ci_population": (psi - 1.96 * sp, psi + 1.96 * sp),
        "ci_sample": (psi - 1.96 * ss, psi + 1.96 * ss),
        "width_ratio": ss / sp if sp > 0 else float("nan"),
        "method": "TMLE with sample-effect inference; van der Laan & "
                  "Rose (2018) Chap. 12",
        "note": "same point estimate; the SAMPLE interval is narrower "
                "by the variance of the conditional effect, and is "
                "asymptotically conservative for the SATE",
    })


def paired_variance(pair_ids, ic):
    r"""Pair-matched variance: the pair is the unit of independence.

    Ignoring the matching and treating units as independent is the
    error this exists to prevent.
    """
    p = list(pair_ids)
    v = [float(q) for q in k.vec(ic)]
    if len(p) != len(v):
        raise ValueError("tlsate: %d pair labels for %d influence "
                         "values" % (len(p), len(v)))
    agg = {}
    for i in range(len(p)):
        agg.setdefault(p[i], []).append(v[i])
    if any(len(q) != 2 for q in agg.values()):
        raise ValueError("tlsate: every pair must contain exactly 2 "
                         "units")
    sums = [sum(q) / 2.0 for q in agg.values()]
    m = sum(sums) / len(sums)
    var = sum((q - m) ** 2 for q in sums) / (len(sums) - 1)
    return {"se": math.sqrt(var / len(sums)), "n_pairs": len(sums),
            "note": "the PAIR is the independent unit"}


def cheatsheet():
    return ("tlsate: in a cluster randomized trial the units are not "
            "sampled from any defined population, so the PATE is a "
            "parameter of a superpopulation nobody drew from. The "
            "SATE -- the mean counterfactual difference for THESE "
            "units -- is interpretable without inventing one. It is "
            "not identifiable in finite samples, but the SAME TMLE is "
            "consistent and asymptotically linear for it; only the "
            "influence curve changes, dropping Q1 - Q0 - psi. The "
            "variance falls by EXACTLY the variance of the conditional "
            "effect, so effect modification is where the power gain "
            "comes from.")


# compact alias per ledger/NAMING.md
sampleate = sate_tmle
