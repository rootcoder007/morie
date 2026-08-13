# morie.fn -- function file (rootcoder007/morie)
r"""Time-dependent covariate adjustment.

Hernan & Robins (2020) Ch. 20 and Sec. 21.2. The chapter's point is a
negative one, and it is what this module is built around: when a
time-varying covariate :math:`L_k` both confounds later treatment and
is itself affected by earlier treatment, **no conditioning strategy
works**. Adjust for it and you block the part of the effect that runs
through it; leave it out and the later treatment stays confounded. The
covariate is a confounder and a mediator at once.

Weighting escapes the trap. The IP weights of Sec. 21.2,

.. math:: SW^{\bar A} = \prod_{k}
          \frac{f\!\left(A_k \mid \bar A_{k-1}\right)}
               {f\!\left(A_k \mid \bar A_{k-1}, \bar L_k\right)},

create a pseudo-population in which :math:`\bar L` no longer predicts
treatment, so a marginal structural model fitted there estimates the
effect of the whole treatment history without ever conditioning on
:math:`L`.

So this function does not return one number. It returns three, and the
comparison is the analysis:

``msm``
    The IP-weighted marginal structural model. The estimate.
``adjusted``
    The same outcome model with :math:`\bar L` entered as regressors --
    the naive fix, biased through over-adjustment.
``unadjusted``
    No adjustment at all, biased through confounding.

Ch. 20 predicts the two naive estimators fall on *opposite sides* of
the truth when the treatment effect and the confounding run the same
way, and the anchor checks exactly that. Reporting only the first
number would hide the finding the chapter exists to make.

References
----------
Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*, Boca
Raton: Chapman & Hall/CRC. Ch. 20 (treatment-confounder feedback),
Sec. 21.2 (IP weighting for time-varying treatments), Sec. 12.3 (the
stabilized weights and the mean-1 diagnostic).

Robins, J. (1986) "A new approach to causal inference in mortality
studies with a sustained exposure period", *Mathematical Modelling*
7(9-12), 1393-1512, doi:10.1016/0270-0255(86)90088-6 -- where the
problem and the weighted solution were first set out.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["time_dep_covariate"]

_CONTRASTS = ("cumulative", "final", "everexposed")


def time_dep_covariate(y, A, L_t, time=None, contrast="cumulative",
                       kind="binary", stabilize=True, trim=None):
    r"""IP-weighted MSM for a time-varying treatment, with the two
    naive comparators alongside it.

    Parameters
    ----------
    y : array-like
        End-of-follow-up outcome.
    A : list of array-like
        Treatment at each time, ``A[k][i]``.
    L_t : list of array-like
        Time-varying covariates, ``L_t[k]`` an n-by-p block.
    time : list, optional
        Labels for the time points, carried through to the result.
        Purely descriptive; the ordering used is the list order.
    contrast : {"cumulative", "final", "everexposed"}
        The exposure summary the marginal structural model regresses on.

    Returns
    -------
    RichResult
        ``estimate`` is the MSM coefficient; ``adjusted`` and
        ``unadjusted`` are the two biased comparators, reported so the
        over-adjustment can be seen rather than assumed away.

    Examples
    --------
    Two periods with feedback from treatment to the covariate::

        r = time_dep_covariate(y, [A0, A1], [L0, L1])
        r["estimate"], r["adjusted"], r["unadjusted"]
    """
    if contrast not in _CONTRASTS:
        raise ValueError("time_dep_covariate: contrast must be one of %r, "
                         "got %r" % (_CONTRASTS, contrast))
    A_hist = list(A) if isinstance(A, (list, tuple)) and A and isinstance(
        A[0], (list, tuple)) else [A]
    L_hist = list(L_t) if isinstance(L_t, (list, tuple)) and (
        not L_t or isinstance(L_t[0], (list, tuple)) or L_t[0] is None
    ) else [L_t]
    K = len(A_hist)
    if K == 0:
        raise ValueError("time_dep_covariate: need at least one time point")
    if len(L_hist) != K:
        raise ValueError(
            "time_dep_covariate: %d treatment times but %d covariate "
            "blocks; Sec. 21.2 needs L-bar_k at every k" % (K, len(L_hist)))
    yv = k.vec(y)
    n = len(yv)
    for kk in range(K):
        if len(k.vec(A_hist[kk])) != n:
            raise ValueError("time_dep_covariate: outcome has %d rows but "
                             "treatment at time %d has %d"
                             % (n, kk, len(k.vec(A_hist[kk]))))

    w, per_time = k.ip_weights_history(A_hist, L_hist, kind=kind,
                                       stabilize=stabilize, trim=trim)

    cum = [sum(k.vec(A_hist[kk])[i] for kk in range(K)) for i in range(n)]
    if contrast == "cumulative":
        expo = cum
    elif contrast == "final":
        expo = list(k.vec(A_hist[-1]))
    else:
        expo = [1.0 if cum[i] > 0.0 else 0.0 for i in range(n)]
    X = [[expo[i]] for i in range(n)]

    msm = k.wls(X, yv, w)
    unadj = k.wls(X, yv, [1.0] * n)

    # the naive fix: put L-bar in the outcome model
    Lcols = []
    for kk in range(K):
        if L_hist[kk] is None:
            continue
        block = k.mat(L_hist[kk])
        for c in range(len(block[0])):
            Lcols.append([row[c] for row in block])
    if Lcols:
        Xadj = [[expo[i]] + [float(c[i]) for c in Lcols] for i in range(n)]
    else:
        Xadj = X
    adj = k.wls(Xadj, yv, [1.0] * n)

    s1 = sum(w)
    s2 = sum(v * v for v in w)
    return RichResult(payload={
        "estimate": msm["coef"][1],
        "se": msm["se"][1],
        "msm": msm["coef"][1],
        "msm_se": msm["se"][1],
        "adjusted": adj["coef"][1],
        "adjusted_se": adj["se"][1],
        "unadjusted": unadj["coef"][1],
        "unadjusted_se": unadj["se"][1],
        "coef": msm["coef"],
        "vcov": msm["vcov"],
        "weights": w,
        "mean_weight": s1 / n,
        "max_weight": max(w),
        "effective_sample_size": (s1 * s1 / s2) if s2 > 0.0 else 0.0,
        "cumulative_exposure": cum,
        "exposure": expo,
        "per_time": [{"time": (time[t] if time is not None
                               and t < len(time) else t),
                      "mean_weight": sum(p["weight"]) / n}
                     for t, p in enumerate(per_time)],
        "n_times": K, "n": n, "contrast": contrast,
        "method": "IP-weighted MSM for a time-varying treatment, "
                  "Hernan & Robins (2020) Sec. 21.2, with the "
                  "over-adjusted and unadjusted comparators of Ch. 20",
    })


def cheatsheet():
    return ("tdcvar: time-varying IPTW MSM (H&R Ch.21). Returns the "
            "weighted MSM plus the two biased comparators -- adjusting "
            "for a treatment-affected confounder over-adjusts, omitting "
            "it under-adjusts, and Ch.20 says they straddle the truth.")


# compact alias per ledger/NAMING.md
timedepcovariate = time_dep_covariate
