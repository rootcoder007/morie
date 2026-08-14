# morie.fn -- function file (rootcoder007/morie)
r"""Data-adaptive target parameters and CV-TMLE.

Sometimes the question itself depends on the data: which exposure
levels to contrast, which of a thousand variables to report an effect
for. Defining the parameter after looking, then estimating it on the
same data, is the mistake that makes reported confidence intervals
meaningless -- and it is not fixed by a better estimator.

**The construction that makes it legitimate.** Split the sample. On
the *training* split, use the data to **define** the parameter --
choose the levels, screen the variables -- producing a
:math:`v`-specific parameter :math:`\Psi_v`. On the *estimation*
split, estimate that now-fixed parameter. Conditional on the training
split the target is a fixed functional, so the usual asymptotics
apply, and the reported parameter is honestly the one that was
estimated.

**Then average across splits, not before.** Each fold gives its own
:math:`\Psi_v` and its own TMLE. CV-TMLE combines them:

.. math:: \psi_n = \frac{1}{V}\sum_{v} \Psi_v(P^*_{n,v}),

with the influence curve pooled over folds. This is not only a
variance reduction -- it is what removes the Donsker condition, since
each fold's fit is independent of the data it is evaluated on.

**A variable importance measure is the natural application.** For each
variable, the parameter is the effect of shifting it with the others
held fixed; screening chooses which variables to report; and CV-TMLE
supplies inference for the reported ones. The anchor exploits the
cost: reusing one sample to both select and estimate inflates the
apparent effect of a null variable, while the split does not.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 9
(data-adaptive target parameters; the example of defining treatment or
exposure levels from the data; methodology for data-adaptive
parameters; the TMLE of the v-specific data-adaptive parameter and the
combination of v-specific TMLEs across estimation samples; CV-TMLE and
CV-TMLE for data-adaptive parameters; CV-TMLE for variable importance
measures; and the varImpact software with the Framingham Heart Study
analysis).

Hubbard, A. E., Kherad-Pajouh, S. & van der Laan, M. J. (2016)
"Statistical Inference for Data Adaptive Target Parameters",
*International Journal of Biostatistics* 12(1), 3-19,
doi:10.1515/ijb-2015-0013.

Zheng, W. & van der Laan, M. J. (2011) "Cross-Validated Targeted
Minimum-Loss-Based Estimation", in *Targeted Learning*, Springer,
459-474, doi:10.1007/978-1-4419-9782-1_27.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["split_sample", "data_adaptive_parameter", "cv_tmle",
           "variable_importance", "naive_reuse"]

_EPS = 1e-12


def split_sample(n, V=10, seed=0):
    r"""Training splits define the parameter; estimation splits
    estimate it."""
    if int(V) < 2 or int(V) > int(n):
        raise ValueError("tldapar: V must lie in 2..%d, got %d"
                         % (n, V))
    rng = np.random.default_rng(seed)
    idx = list(range(int(n)))
    for i in range(len(idx) - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    est = [sorted(idx[v::int(V)]) for v in range(int(V))]
    return {"estimation": est,
            "training": [sorted(set(range(int(n))) - set(e))
                         for e in est], "V": int(V)}


def data_adaptive_parameter(define_on_training, estimate_on_holdout,
                            n, V=10, seed=0):
    r"""Define on the training split, estimate on the held-out one.

    ``define_on_training(train_idx)`` returns the fold-specific
    parameter (any object); ``estimate_on_holdout(param, est_idx)``
    returns its estimate and influence-curve values.
    """
    sp = split_sample(n, V, seed)
    ests, ics, params = [], [0.0] * int(n), []
    for v in range(sp["V"]):
        p = define_on_training(sp["training"][v])
        params.append(p)
        r = estimate_on_holdout(p, sp["estimation"][v])
        ests.append(float(r["estimate"]))
        for a, i in enumerate(sp["estimation"][v]):
            ics[i] = float(r["ic"][a])
    psi = sum(ests) / len(ests)
    m = sum(ics) / len(ics)
    se = math.sqrt(sum((v - m) ** 2 for v in ics)
                   / (len(ics) - 1) / len(ics))
    return RichResult(payload={
        "estimate": psi, "psi": psi, "fold_estimates": ests,
        "fold_parameters": params, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "V": sp["V"],
        "method": "data-adaptive target parameter with CV-TMLE; van "
                  "der Laan & Rose (2018) Chap. 9",
        "note": "the parameter is FIXED conditional on the training "
                "split, so the reported quantity is the one that was "
                "estimated",
    })


def cv_tmle(fold_estimates, fold_ics, n):
    r"""Combine :math:`v`-specific TMLEs and pool the influence
    curve."""
    e = [float(v) for v in k.vec(fold_estimates)]
    if not e:
        raise ValueError("tldapar: no fold estimates given")
    psi = sum(e) / len(e)
    ic = []
    for f in fold_ics:
        ic.extend(float(v) for v in k.vec(f))
    if len(ic) != int(n):
        raise ValueError("tldapar: %d influence-curve values for %d "
                         "observations" % (len(ic), n))
    m = sum(ic) / len(ic)
    se = math.sqrt(sum((v - m) ** 2 for v in ic)
                   / (len(ic) - 1) / len(ic))
    return {"psi": psi, "se": se,
            "ci": (psi - 1.96 * se, psi + 1.96 * se),
            "mean_ic": m,
            "note": "each fold's fit is independent of the data it is "
                    "evaluated on, which is what removes the Donsker "
                    "condition"}


def variable_importance(X, Y, screen, effect, V=5, seed=0):
    r"""CV-TMLE variable importance.

    ``screen(train_idx)`` selects which variables to report;
    ``effect(j, est_idx)`` estimates variable :math:`j`'s effect with
    its influence curve. Only variables selected in a fold are
    estimated in that fold, which is the whole point.
    """
    rows = [[float(v) for v in r] for r in k.mat(X)]
    n = len(rows)
    sp = split_sample(n, V, seed)
    per = {}
    for v in range(sp["V"]):
        sel = screen(sp["training"][v])
        for j in sel:
            r = effect(j, sp["estimation"][v])
            per.setdefault(j, {"est": [], "ic": []})
            per[j]["est"].append(float(r["estimate"]))
            per[j]["ic"].extend(float(q) for q in r["ic"])
    out = {}
    for j, d in per.items():
        psi = sum(d["est"]) / len(d["est"])
        ic = d["ic"]
        m = sum(ic) / len(ic)
        se = math.sqrt(sum((q - m) ** 2 for q in ic)
                       / max(len(ic) - 1, 1) / len(ic))
        out[j] = {"psi": psi, "se": se,
                  "ci": (psi - 1.96 * se, psi + 1.96 * se),
                  "folds_selected": len(d["est"])}
    return RichResult(payload={
        "estimate": out, "importance": out, "V": sp["V"],
        "method": "CV-TMLE variable importance; van der Laan & Rose "
                  "(2018) Chap. 9, as in varImpact",
    })


def naive_reuse(define_and_estimate, n, seed=0):
    r"""The mistake, implemented so its cost is measurable.

    Defines the parameter and estimates it on the SAME data. Kept so
    the anchor can show the inflation rather than assert it.
    """
    r = define_and_estimate(list(range(int(n))))
    return {"estimate": float(r["estimate"]),
            "warning": "the parameter was selected and estimated on "
                       "the same sample; the reported inference is "
                       "not valid for the selected parameter"}


def cheatsheet():
    return ("tldapar: when the QUESTION depends on the data -- which "
            "levels to contrast, which variable to report -- defining "
            "and estimating on the same sample invalidates the "
            "interval, and no estimator fixes that. Split: DEFINE the "
            "parameter on the training split, ESTIMATE it on the "
            "held-out one, so conditional on training it is fixed. "
            "Then CV-TMLE averages the v-specific TMLEs and pools the "
            "influence curve -- which also removes the Donsker "
            "condition, since each fit is independent of the data it "
            "is scored on.")


# compact alias per ledger/NAMING.md
dataadaptiveparameter = data_adaptive_parameter
