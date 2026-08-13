# morie.fn -- function file (rootcoder007/morie)
r"""Data-adaptive target parameters: honest inference after snooping.

A variable importance question -- *which* exposure level matters most --
often has no pre-specified answer. There is no objectively defined
high-risk and low-risk level of blood pressure or cholesterol, so the
levels have to be found in the data:

.. math::
   a_L(P_n) = \arg\min_{a} \frac1n \sum_i Q_n(a, W_i),
   \qquad
   a_H(P_n) = \arg\max_{a} \frac1n \sum_i Q_n(a, W_i),

and the parameter of interest is then the contrast between them,

.. math:: \Psi_{a_L, a_H}(P) = E_W\{E(Y \mid A=a_H, W)
                                    - E(Y \mid A=a_L, W)\}.

**The dual use of the data is the whole problem.** Pick the levels and
estimate the contrast on the same rows and the substitution estimator
(9.5) is biased upward -- and not a little. Under a null where the
exposure does nothing at all, the estimator is *always* non-negative,
because it is the max minus the min of the same noisy surface. It is a
winner's curse with a confidence interval attached. The anchor builds
exactly that null and checks the naive arm is positive in every single
replicate, because a claim about a bias that is structurally one-sided
should be checked as one.

**The fix is a split, and which split you use costs you something
different.**

``"sample-split"``
    Sec. 9.3-9.4. The parameter-generating sample :math:`P_{n,v^c}`
    supplies :math:`(a_L, a_H)`; a separate estimation sample
    :math:`P_{n,v}` fits Q and g and does the targeting. Honest, but the
    nuisances see only :math:`n/V` rows.

``"cv-tmle"``
    Sec. 9.5-9.6, the default. The nuisances are fitted on the
    parameter-generating sample too, and *only* the one-dimensional
    fluctuation :math:`\epsilon` is fitted on the estimation sample.
    Same honesty, far more data behind Q and g, and no Donsker
    condition restricting how adaptive the level-finder is allowed to
    be.

Both average the split-specific estimates,
:math:`\hat\Psi(P_n) = \frac1V \sum_v \hat\Psi^{TMLE}_{P_{n,v^c}}(P_{n,v})`,
and take the variance from the average of the split-specific influence
curves, eq. (9.15).

**Near a tie the interval is not to be trusted, and it says so.** The
levels are an argmin and an argmax, so when two levels are genuinely
tied the index :math:`\hat d` is not unique and the parameter is
non-regular -- the same difficulty the "exceptional law" condition names
elsewhere in this volume. Measured under an exact null where all levels
tie, the split arms stay unbiased but the influence-curve interval
under-covers, and *worsens* with n (95 percent nominal delivering about
80 percent at n = 400 and about 63 percent at n = 1600) because the
estimator's own spread stops shrinking at the parametric rate while the
reported standard error keeps shrinking. Where the levels are separated
the interval is well calibrated (ratio of reported to actual spread 1.03
at n = 600, coverage 29 of 30 at n = 1200). ``separation`` and
``level_agreement`` are returned so a caller can see which regime they
are in rather than discovering it as a coverage failure.

**The estimand moves with the data, and that is deliberate.** What is
being estimated is :math:`\frac1V \sum_v \Psi_{\hat d(P_{n,v^c})}(P_0)`
-- the average, over splits, of the true contrast at the levels that
split happened to pick. It is a property of :math:`P_0` for a
data-chosen index, not a fixed parameter, and the confidence interval
covers *that*.

References
----------
Hubbard, A. E., Kennedy, C. J. & van der Laan, M. J. (2018)
"Data-Adaptive Target Parameters", Ch. 9 in van der Laan, M. J. &
Rose, S. (eds.) *Targeted Learning in Data Science: Causal Inference for
Complex Longitudinal Studies*, Springer Series in Statistics,
pp. 125-142, doi:10.1007/978-3-319-65304-4_9. Eq. (9.1)-(9.16).

Hubbard, A. E., Kherad-Pajouh, S. & van der Laan, M. J. (2016)
"Statistical Inference for Data Adaptive Target Parameters", *The
International Journal of Biostatistics* 12(1), 3-19,
doi:10.1515/ijb-2015-0013. The sample-splitting theory the chapter
builds on.

van der Laan, M. J. & Luedtke, A. R. (2015) "Targeted Learning of the
Mean Outcome Under an Optimal Dynamic Treatment Rule", *Journal of
Causal Inference* 3(1), 61-95, doi:10.1515/jci-2013-0022. The CV-TMLE
for data-adaptive parameters of Sec. 9.6.

Zheng, W. & van der Laan, M. J. (2011) "Cross-Validated Targeted
Minimum-Loss-Based Estimation", in van der Laan, M. J. & Rose, S.
*Targeted Learning*, Springer Series in Statistics, pp. 459-474,
doi:10.1007/978-1-4419-9782-1_27.

Wilson, P. W. F., D'Agostino, R. B., Levy, D., Belanger, A. M.,
Silbershatz, H. & Kannel, W. B. (1998) "Prediction of Coronary Heart
Disease Using Risk Factor Categories", *Circulation* 97(18), 1837-1847,
doi:10.1161/01.CIR.97.18.1837. The Framingham analysis the chapter
compares against.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tmle_data_adaptive", "discover_levels", "variable_importance",
           "split_specific_tmle"]

_METHODS = ("cv-tmle", "sample-split", "naive")
_EPS = 1e-9


def _logit(p):
    q = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(q / (1.0 - q))


def _levels(A, candidate_strata):
    if candidate_strata is not None:
        lv = [float(v) for v in candidate_strata]
        seen = set()
        lv = [v for v in lv if not (v in seen or seen.add(v))]
    else:
        lv = sorted(set(float(v) for v in A))
    if len(lv) < 2:
        raise ValueError("tmldta: need at least 2 exposure levels, got "
                         "%d" % len(lv))
    return lv


def _fit_q(y, A, W, levels, rows, ridge):
    """Q(a, W) = E[Y | A = a, W], with a dummy per level and level-by-W
    interactions so the surface is free to differ in shape by level --
    without them every level would share one slope and the argmax could
    not move with W."""
    ref = levels[0]
    others = levels[1:]
    p = len(W[0]) if W and W[0] else 0

    def row(a, i):
        d = [1.0 if a == lv else 0.0 for lv in others]
        r = [1.0] + d + list(W[i])
        for t in range(len(others)):
            r += [d[t] * W[i][c] for c in range(p)]
        return r

    X = [row(A[i], i) for i in rows]
    b = k.logit_irls(X, [y[i] for i in rows], ridge=max(ridge, 1e-10))

    def q(a, i):
        r = row(a, i)
        return k.sigmoid(sum(b[t] * r[t] for t in range(len(b))))

    return q, b


def _fit_g(A, W, aL, aH, rows, ridge, trim):
    r"""g(a | W) for the two levels that matter.

    Sec. 9.3's second route: collapse A to A* = A when
    A in {aL, aH} and a third value otherwise, fit each category by
    binary logistic regression, then normalise so the three are a proper
    conditional distribution. Only two of the three are ever used, but
    normalising is what makes them probabilities rather than three
    unrelated fits.
    """
    n = len(A)
    X = k.design(W, n)
    Xr = [X[i] for i in rows]

    def cat(mask):
        b = k.logit_irls(Xr, [mask[i] for i in rows],
                         ridge=max(ridge, 1e-10))
        return [k.sigmoid(v) for v in k.matvec(X, b)]

    pH = cat([1.0 if A[i] == aH else 0.0 for i in range(n)])
    pL = cat([1.0 if A[i] == aL else 0.0 for i in range(n)])
    pO = cat([0.0 if A[i] in (aH, aL) else 1.0 for i in range(n)])
    gH, gL = [], []
    for i in range(n):
        tot = pH[i] + pL[i] + pO[i]
        if tot <= 0.0:
            gH.append(0.5)
            gL.append(0.5)
            continue
        gH.append(min(max(pH[i] / tot, trim), 1.0 - trim))
        gL.append(min(max(pL[i] / tot, trim), 1.0 - trim))
    return gH, gL


def discover_levels(y, A, W, levels, rows=None, eval_rows=None,
                    ridge=1e-8):
    r"""Eq. (9.2)-(9.3): the levels that minimise and maximise the mean
    predicted outcome.

    ``rows`` fits Q; ``eval_rows`` supplies the covariates the mean is
    taken over. Keeping them separate is what lets the caller find the
    levels on the parameter-generating sample only.
    """
    n = len(y)
    rows = list(range(n)) if rows is None else list(rows)
    eval_rows = rows if eval_rows is None else list(eval_rows)
    q, _ = _fit_q(y, A, W, levels, rows, ridge)
    means = {}
    for a in levels:
        means[a] = sum(q(a, i) for i in eval_rows) / len(eval_rows)
    aL = min(levels, key=lambda a: means[a])
    aH = max(levels, key=lambda a: means[a])
    return aL, aH, {"means": means, "spread": means[aH] - means[aL]}


def split_specific_tmle(y, A, W, levels, aL, aH, fit_rows, est_rows,
                        ridge=1e-8, trim=0.01, target=True):
    r"""Eq. (9.9)-(9.13): one split's TMLE at fixed levels.

    ``fit_rows`` fits Q and g; ``est_rows`` fits the fluctuation and
    supplies the covariate distribution averaged over. Passing the same
    rows for both is the sample-split arm; passing the
    parameter-generating rows to ``fit_rows`` is the CV-TMLE arm.
    """
    n = len(y)
    q, _ = _fit_q(y, A, W, levels, fit_rows, ridge)
    gH, gL = _fit_g(A, W, aL, aH, fit_rows, ridge, trim)
    H = [(1.0 / gH[i] if A[i] == aH else 0.0)
         - (1.0 / gL[i] if A[i] == aL else 0.0) for i in range(n)]
    off = [_logit(q(A[i], i)) for i in range(n)]
    eps = (k.logistic_fluctuation(y, off, H, est_rows) if target
           else 0.0)

    def qstar(a, i):
        h = (1.0 / gH[i]) if a == aH else (-1.0 / gL[i])
        return k.sigmoid(_logit(q(a, i)) + eps * h)

    m = len(est_rows)
    psi = sum(qstar(aH, i) - qstar(aL, i) for i in est_rows) / m
    D = {}
    for i in est_rows:
        resid = k.sigmoid(off[i] + eps * H[i])
        D[i] = (H[i] * (y[i] - resid)
                + qstar(aH, i) - qstar(aL, i) - psi)
    return psi, D, {"eps": eps, "max_weight": max(
        max(1.0 / gH[i], 1.0 / gL[i]) for i in est_rows)}


def _folds(n, n_folds):
    V = max(2, min(int(n_folds), n))
    return [[i for i in range(n) if i % V == v] for v in range(V)]


def tmle_data_adaptive(y, D, X, candidate_strata=None,
                       method="cv-tmle", n_folds=10, trim=0.01,
                       ridge=1e-8, level=0.95, bounds=None):
    r"""Contrast between data-discovered exposure levels, done honestly.

    Parameters
    ----------
    y : array-like
        Outcome. Rescaled into [0, 1] for the logistic submodel and
        reported on the original scale.
    D : array-like
        The exposure whose importance is in question, discrete.
    X : array-like
        The remaining covariates W.
    candidate_strata : array-like, optional
        The levels of ``D`` to search over. Defaults to its sorted
        distinct values.
    method : {"cv-tmle", "sample-split", "naive"}
        ``cv-tmle`` (Sec. 9.6, the default) fits Q and g on the
        parameter-generating sample and only the fluctuation on the
        estimation sample. ``sample-split`` (Sec. 9.3-9.4) fits
        everything on the estimation sample. ``naive`` is the
        substitution estimator (9.5) with no split at all -- kept
        because its bias is the point of the chapter, and a bias you
        can measure is better than one you assume away.
    n_folds : int
        V in the V-fold split.

    Returns
    -------
    RichResult
        ``estimate`` with ``se`` and ``ci``, the levels each split
        chose, and the per-split estimates.

    Examples
    --------
    Rank a variable's importance without pre-specifying its levels::

        r = tmle_data_adaptive(y, A, W)
        r["estimate"], r["ci"], r["levels_by_split"]
    """
    if method not in _METHODS:
        raise ValueError("tmldta: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    yv, Av = k.vec(y), k.vec(D)
    n = len(yv)
    if len(Av) != n:
        raise ValueError("tmldta: %d outcomes but %d exposures"
                         % (n, len(Av)))
    Wm = k.mat(X) if X is not None else [[] for _ in range(n)]
    if len(Wm) != n:
        raise ValueError("tmldta: %d covariate rows for %d outcomes"
                         % (len(Wm), n))
    if not 0.0 < float(trim) < 0.5:
        raise ValueError("tmldta: trim must be in (0, 0.5), got %r"
                         % (trim,))
    if n < 8:
        raise ValueError("tmldta: need at least 8 observations, got %d"
                         % n)
    lv = _levels(Av, candidate_strata)
    missing = [a for a in lv if not any(v == a for v in Av)]
    if missing:
        raise ValueError("tmldta: candidate levels %s never occur"
                         % (missing,))

    lo, hi = (min(yv), max(yv)) if bounds is None else (float(bounds[0]),
                                                        float(bounds[1]))
    rng = hi - lo
    if rng <= 0.0:
        raise ValueError("tmldta: the outcome has no range")
    if any(v < lo - 1e-12 or v > hi + 1e-12 for v in yv):
        raise ValueError("tmldta: an outcome falls outside bounds")
    ys = [min(max((v - lo) / rng, 0.0), 1.0) for v in yv]

    all_rows = list(range(n))
    if method == "naive":
        # eq. (9.5): the same rows choose the levels AND estimate the
        # contrast. Structurally >= 0 under the null.
        aL, aH, dinfo = discover_levels(ys, Av, Wm, lv, all_rows,
                                        ridge=ridge)
        psi, Dic, info = split_specific_tmle(ys, Av, Wm, lv, aL, aH,
                                             all_rows, all_rows, ridge,
                                             trim, target=False)
        splits = [{"aL": aL, "aH": aH, "estimate": rng * psi,
                   "n_est": n}]
        sigma2 = sum(v * v for v in Dic.values()) / n
        psi_hat, eps_all = psi, [0.0]
    else:
        folds = _folds(n, n_folds)
        splits, per_split, ics, eps_all = [], [], [], []
        for est in folds:
            gen = [i for i in all_rows if i not in set(est)]
            if not gen or not est:
                continue
            # the levels come from the parameter-generating sample only
            aL, aH, _ = discover_levels(ys, Av, Wm, lv, gen,
                                        ridge=ridge)
            fit = gen if method == "cv-tmle" else est
            psi_v, D_v, info_v = split_specific_tmle(
                ys, Av, Wm, lv, aL, aH, fit, est, ridge, trim)
            per_split.append(psi_v)
            eps_all.append(info_v["eps"])
            ics.append([D_v[i] for i in est])
            splits.append({"aL": aL, "aH": aH,
                           "estimate": rng * psi_v, "n_est": len(est)})
        if not per_split:
            raise ValueError("tmldta: no usable splits")
        psi_hat = sum(per_split) / len(per_split)          # eq. (9.14)
        # eq. (9.15): average of the split-specific IC variances
        sigma2 = sum(sum(v * v for v in ic) / len(ic)
                     for ic in ics) / len(ics)

    psi = rng * psi_hat
    se = rng * math.sqrt(sigma2 / n)
    z = k.qnorm(0.5 + 0.5 * float(level))
    chosen = {}
    for sp in splits:
        key = (sp["aL"], sp["aH"])
        chosen[key] = chosen.get(key, 0) + 1
    modal = max(chosen, key=lambda kk: chosen[kk])
    agreement = chosen[modal] / float(len(splits))
    # how far the chosen levels stand clear of their nearest rival.
    # Compared below against the estimate's own standard error, not an
    # absolute cut: under a true tie the ESTIMATED gap is sampling noise
    # of order 1/sqrt(n), so any fixed threshold fires at one sample
    # size and not another.
    _, _, sep_info = discover_levels(ys, Av, Wm, lv, all_rows,
                                     ridge=ridge)
    ordered = sorted(sep_info["means"].values())
    separation = min(ordered[1] - ordered[0],
                     ordered[-1] - ordered[-2]) * rng

    return RichResult(payload={
        "estimate": psi, "se": se, "n": n,
        "ci": (psi - z * se, psi + z * se), "level": float(level),
        "levels_by_split": [(sp["aL"], sp["aH"]) for sp in splits],
        "level_counts": chosen,
        "modal_levels": modal,
        "level_agreement": agreement,
        "separation": separation,
        "near_tie": separation < 2.0 * se or agreement < 0.6,
        "level_means": {a: rng * v + lo
                        for a, v in sep_info["means"].items()},
        "split_estimates": [sp["estimate"] for sp in splits],
        "n_splits": len(splits), "epsilon": eps_all,
        "candidate_levels": lv, "method": method,
        "sigma": math.sqrt(sigma2) * rng,
        "algorithm": "data-adaptive target parameter, Hubbard, Kennedy "
                     "& van der Laan (2018) Ch. 9 eq. (9.2)-(9.16)",
    })


def variable_importance(y, X, candidate_strata=None, method="cv-tmle",
                        n_folds=10, names=None, **kw):
    r"""Loop the contrast over every column of ``X`` in turn.

    For each variable the chapter's construction is applied with that
    column as A and everything else as W, so the results are on a common
    scale and can be ranked. Returned sorted by the absolute estimate,
    largest first.
    """
    Xm = k.mat(X)
    n = len(Xm)
    p = len(Xm[0]) if n else 0
    if p < 2:
        raise ValueError("variable_importance: need at least 2 columns,"
                         " got %d" % p)
    nm = (list(names) if names is not None
          else ["X%d" % (j + 1) for j in range(p)])
    if len(nm) != p:
        raise ValueError("variable_importance: %d names for %d columns"
                         % (len(nm), p))
    out = []
    for j in range(p):
        A = [Xm[i][j] for i in range(n)]
        W = [[Xm[i][c] for c in range(p) if c != j] for i in range(n)]
        r = tmle_data_adaptive(y, A, W, candidate_strata=candidate_strata,
                               method=method, n_folds=n_folds, **kw)
        out.append({"variable": nm[j], "index": j,
                    "estimate": r["estimate"], "se": r["se"],
                    "ci": r["ci"], "levels": r["modal_levels"]})
    out.sort(key=lambda d: -abs(d["estimate"]))
    for rank, d in enumerate(out):
        d["rank"] = rank + 1
    return out


def cheatsheet():
    return ("tmldta: levels found in the data (aL = argmin, aH = argmax "
            "of mean Q(a,W)) then the contrast estimated -- but NOT on "
            "the same rows. Naive reuse is structurally >= 0 under the "
            "null. cv-tmle fits Q and g on the parameter-generating "
            "split and only epsilon on the estimation split; average "
            "the split estimates (9.14), variance from the average of "
            "the split influence curves (9.15).")


# compact alias per ledger/NAMING.md
tmledataadaptive = tmle_data_adaptive
