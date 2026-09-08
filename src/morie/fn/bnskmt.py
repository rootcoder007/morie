# morie.fn -- function file (rootcoder007/morie)
r"""Conditional moment inequalities: the Kolmogorov-Smirnov form.

Companion to :mod:`bndsmw`. Both convert conditional moment
inequalities into infinitely many unconditional ones through weight
functions :math:`g(X)`; they differ in how the resulting family of
standardised moments is collapsed to one number.

**CvM integrates, KS takes the supremum:**

.. math:: T^{CvM}_n(\theta)
            &= \int S\big(n^{1/2}\bar m_n(\theta,g),
               \Sigma_n(\theta,g)\big)\,dQ(g), \\
          T^{KS}_n(\theta)
            &= \sup_{g \in \mathcal G}
               S\big(n^{1/2}\bar m_n(\theta,g),
               \Sigma_n(\theta,g)\big).

**The difference is not cosmetic.** A supremum is driven entirely by
the single worst instrument, so KS responds sharply to a violation
concentrated on one small region of :math:`X` -- a sliver where the
inequality fails badly -- while CvM averages that sliver against every
other instrument and may barely notice it. Conversely a violation
spread thinly across many regions accumulates under CvM and shows up
only weakly under KS.

Neither dominates. The anchor constructs both cases and measures the
reversal rather than asserting a ranking: a violation on one narrow
cell makes the KS statistic large relative to its own null, and a
diffuse violation of the same total magnitude does the opposite.

**A supremum over a countable class is a maximum in practice.** The
class of hypercube indicators is countable, so the supremum is
attained on the truncated class actually used, and the truncation
error is one-sided -- adding instruments can only raise the statistic.
That monotonicity is a property worth checking, since it means a
coarse class gives a conservative test rather than an unreliable one.

**Same critical values, same slackness problem.** GMS applies
unchanged: an inequality slack by more than :math:`\kappa_n` at the
sample size in hand cannot generate a violation, so it is dropped from
the bootstrap. Under KS this matters more, not less, because a single
badly-scaled slack moment would otherwise dominate the supremum.

References
----------
Andrews, D. W. K. & Shi, X. "Inference Based on Conditional Moment
Inequalities", Cowles Foundation Discussion Paper No. 1761R, June
2010, revised July 2011; published as *Econometrica* 81(2), 609-666
(2013), doi:10.3982/ECTA9370. Sec. 1: "The KS statistic is given by a
supremum over g in G. The CvM statistic is given by an integral with
respect to a probability measure Q on the space G of g functions."
Also the class G with no information loss, the Sum/QLR/Max forms of
S, and GMS critical values.

Andrews, D. W. K. & Soares, G. (2010) "Inference for Parameters
Defined by Moment Inequalities Using Generalized Moment Selection",
*Econometrica* 78(1), 119-157, doi:10.3982/ECTA7502. The GMS
construction.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .bndsmw import (S_function, hypercube_instruments,
                     weighted_moments)

__all__ = ["ks_statistic", "ks_critical_value", "ks_confidence_set",
           "compare_forms"]

_EPS = 1e-12


def ks_statistic(m, instruments, form="sum", n_equality=0):
    r"""The supremum of :math:`S` over the instrument class.

    Also reports which instrument attained it, since under KS the
    answer is a statement about one region of :math:`X` and that is
    usually the interesting part.
    """
    G = instruments["instruments"] if isinstance(instruments, dict) \
        else instruments
    if not G:
        raise ValueError("bnskmt: the instrument class is empty")
    best, arg, parts = 0.0, None, []
    for a, g in enumerate(G):
        wm = weighted_moments(m, g)
        n = wm["n"]
        std = [math.sqrt(n) * wm["mean"][j] / max(wm["sd"][j], _EPS)
               for j in range(len(wm["mean"]))]
        s = S_function(std, form=form, n_equality=n_equality)
        parts.append(s)
        if s > best:
            best, arg = s, a
    return {"statistic": best, "argmax": arg,
            "per_instrument": parts, "form": form,
            "n_instruments": len(G),
            "method": "Kolmogorov-Smirnov: supremum of S over G "
                      "(Andrews & Shi, Sec. 1)"}


def ks_critical_value(m, instruments, form="sum", n_equality=0,
                      level=0.95, reps=200, seed=0, kappa=None):
    r"""GMS critical value for the supremum statistic."""
    M = [[float(v) for v in r] for r in k.mat(m)]
    n = len(M)
    G = instruments["instruments"] if isinstance(instruments, dict) \
        else instruments
    kap = float(kappa) if kappa is not None \
        else math.sqrt(math.log(max(n, 3)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(reps)):
        idx = [int(float(rng.uniform()) * n) % n for _ in range(n)]
        Mb = [M[i] for i in idx]
        best = 0.0
        for g in G:
            gb = [g[i] for i in idx]
            wm = weighted_moments(Mb, gb)
            wm0 = weighted_moments(M, g)
            std = []
            for j in range(len(wm["mean"])):
                sd = max(wm0["sd"][j], _EPS)
                xi = math.sqrt(n) * wm0["mean"][j] / sd
                centred = math.sqrt(n) * (wm["mean"][j]
                                          - wm0["mean"][j]) / sd
                std.append(centred + (0.0 if xi <= kap else 1e6))
            s = S_function(std, form=form, n_equality=n_equality)
            best = max(best, s)
        draws.append(best)
    draws.sort()
    q = draws[min(len(draws) - 1, int(float(level) * len(draws)))]
    return {"critical_value": q, "kappa": kap, "reps": int(reps),
            "level": float(level)}


def ks_confidence_set(moment_fn, theta_grid, X, form="sum",
                      n_equality=0, level=0.95, n_levels=2,
                      reps=100, seed=0):
    r"""Invert the KS test over a grid."""
    inst = hypercube_instruments(X, n_levels=n_levels)
    keep, stats = [], {}
    for th in theta_grid:
        m = moment_fn(th)
        t = ks_statistic(m, inst, form=form, n_equality=n_equality)
        c = ks_critical_value(m, inst, form=form,
                              n_equality=n_equality, level=level,
                              reps=reps, seed=seed)
        stats[th] = (t["statistic"], c["critical_value"])
        if t["statistic"] <= c["critical_value"]:
            keep.append(th)
    return RichResult(payload={
        "estimate": keep, "set": keep, "n_in_set": len(keep),
        "bounds": (min(keep), max(keep)) if keep else None,
        "statistics": stats, "form": form, "level": float(level),
        "n_instruments": inst["n_instruments"],
        "method": "KS test with GMS critical values, inverted over "
                  "the grid; Andrews & Shi",
    })


def compare_forms(m, instruments, form="sum", n_equality=0):
    r"""CvM and KS side by side on the same moments.

    The supremum is never below the average, so the ratio is at least
    one; how far above depends on whether the violation is
    concentrated or diffuse.
    """
    from .bndsmw import cvm_statistic
    cv = cvm_statistic(m, instruments, form=form,
                       n_equality=n_equality)
    ks = ks_statistic(m, instruments, form=form,
                      n_equality=n_equality)
    return {"cvm": cv["statistic"], "ks": ks["statistic"],
            "ratio_ks_over_cvm": ks["statistic"]
            / max(cv["statistic"], _EPS),
            "argmax_instrument": ks["argmax"],
            "note": "KS is driven by the single worst instrument, CvM "
                    "by the average; a concentrated violation favours "
                    "KS and a diffuse one favours CvM"}


def cheatsheet():
    return ("bnskmt: conditional moment inequalities, KS form. Same "
            "construction as bndsmw -- conditional inequality becomes "
            "E[m g(X)] >= 0 for all non-negative g -- but the family "
            "is collapsed by a SUPREMUM over g rather than an "
            "integral against Q. KS therefore reacts to a violation "
            "concentrated on ONE region of X, where CvM averages it "
            "away; a diffuse violation reverses that. Adding "
            "instruments can only RAISE the supremum, so truncation "
            "is conservative.")


# compact alias per ledger/NAMING.md
kernelmomentbound = ks_confidence_set

# public names resolved by fn/_lazy_map.json
bound_kernel_moment = ks_confidence_set
