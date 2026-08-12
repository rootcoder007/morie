r"""Synthetic control estimates and placebo (permutation) inference.

Abadie, A., Diamond, A., & Hainmueller, J. (2015) "Comparative Politics and
the Synthetic Control Method", *American Journal of Political Science* 59(2),
495-510.

The synthetic control for a treated unit is a weighted average of the donor
pool, with weights :math:`W = (w_2, \dots, w_{J+1})'` satisfying
:math:`0 \le w_j \le 1` and :math:`\sum_j w_j = 1`, chosen to minimise
:math:`\lVert X_1 - X_0 W \rVert_V` over pre-intervention predictors. The
restriction to the simplex is the whole point of the estimator, and the paper
spends a section on it: regression weights "are not restricted to lie between
zero and one" and "may take on negative values", so a regression counterfactual
extrapolates "beyond the support of comparison units", while the synthetic
control does not.

**Inference is by permutation, not by a standard error.** The paper's
"in-space placebos" reassign the intervention "not in time, but to members of
the donor pool": run the whole procedure pretending each donor was treated,
which "creates a distribution of placebo effects against which we can then
evaluate the effect estimated for the unit that represents the case of
interest". Confidence in a large estimate "would be undermined if the
magnitude of the estimated effect fell well inside the distribution of placebo
effects", and the comparison is operationalised as a p-value: "the fraction of
such effects greater than or equal to the effect estimated for the unit
representing the case of interest".

That fraction rule is what :func:`plcbsc` computes by default. A second
statistic is offered -- the ratio of post- to pre-intervention root mean
squared prediction error, which discounts donors the method fits badly before
the intervention -- and it is labelled as an option rather than attributed:
the 2015 paper describes the placebo design and the fraction rule, not that
ratio.

**In-time placebos** are the other falsification exercise the paper names:
move the intervention date earlier, into a period where nothing happened, and
check that no effect appears. :func:`in_time_placebo` runs it.

One caution the estimator carries and the paper names: when the donors share
a common factor structure, many convex combinations reproduce the treated
unit's pre-intervention path equally well, so the *weights* are not
identified even though the fitted counterfactual is. The paper mentions
penalty terms "when minimization of :math:`\lVert X_1 - X_0 W\rVert` has
multiple solutions"; no penalty is applied here, so read the weights as one
solution among many and the counterfactual as the estimate.

The optimisation over :math:`W` here is projected gradient descent on the
simplex for a fixed :math:`V`; the nested search over :math:`V` (and the
cross-validation the paper uses to choose it) is left to the caller through
the ``v`` argument, which is where the paper itself leaves it in its
methodological section.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["plcbsc", "placebo_inference", "placebo_scm_inference", "synthetic_control",
           "in_time_placebo", "simplex_project"]


def simplex_project(v):
    """Euclidean projection onto the probability simplex."""
    n = len(v)
    u = sorted(v, reverse=True)
    css = 0.0
    rho, theta = 0, 0.0
    for i in range(n):
        css += u[i]
        t = (css - 1.0) / (i + 1)
        if u[i] - t > 0:
            rho, theta = i + 1, t
    return [max(x - theta, 0.0) for x in v]


def synthetic_control(x_treated, x_donors, v=None, max_iter=5000,
                      tol=1e-12, step=None):
    r"""Weights :math:`W` minimising :math:`\lVert X_1 - X_0 W\rVert_V` on
    the simplex.

    Parameters
    ----------
    x_treated : sequence of float
        :math:`X_1`, the treated unit's pre-intervention predictors.
    x_donors : sequence of sequences
        :math:`X_0`, one column per donor: ``x_donors[j]`` is donor
        :math:`j`'s predictor vector.
    v : sequence of float, optional
        The diagonal of :math:`V`, one non-negative weight per predictor.
        Equal weights by default.

    Returns
    -------
    dict
        ``weights``, ``loss``, ``fitted`` (:math:`X_0 W`), ``n_iter``,
        ``converged``.
    """
    X1 = [float(t) for t in x_treated]
    D = [[float(t) for t in col] for col in x_donors]
    k = len(X1)
    J = len(D)
    if J == 0:
        raise ValueError("plcbsc: the donor pool is empty")
    if any(len(col) != k for col in D):
        raise ValueError("plcbsc: every donor needs the same predictors as "
                         "the treated unit")
    if v is None:
        vv = [1.0] * k
    else:
        vv = [float(t) for t in v]
        if len(vv) != k or any(t < 0 for t in vv):
            raise ValueError("plcbsc: v must be one non-negative weight per "
                             "predictor")
        if sum(vv) <= 0:
            raise ValueError("plcbsc: v must have some positive weight")

    w = [1.0 / J] * J

    def resid(ws):
        return [X1[i] - sum(D[j][i] * ws[j] for j in range(J))
                for i in range(k)]

    def loss(ws):
        r = resid(ws)
        return sum(vv[i] * r[i] * r[i] for i in range(k))

    if step is None:
        norm = max(sum(vv[i] * D[j][i] ** 2 for i in range(k))
                   for j in range(J))
        step = 1.0 / (2.0 * norm * J) if norm > 0 else 1e-3
    cur = loss(w)
    converged = False
    it = 0
    for it in range(1, int(max_iter) + 1):
        r = resid(w)
        grad = [-2.0 * sum(vv[i] * r[i] * D[j][i] for i in range(k))
                for j in range(J)]
        s = step
        for _ in range(60):
            cand = simplex_project([w[j] - s * grad[j] for j in range(J)])
            new = loss(cand)
            if new <= cur:
                break
            s *= 0.5
        if new > cur - tol * max(1.0, abs(cur)):
            w, cur = cand, new
            converged = True
            break
        w, cur = cand, new
    return {"weights": w, "loss": cur,
            "fitted": [sum(D[j][i] * w[j] for j in range(J))
                       for i in range(k)],
            "n_iter": it, "converged": converged}


def _gaps(y_treated, y_donors, weights):
    T = len(y_treated)
    return [y_treated[t] - sum(y_donors[j][t] * weights[j]
                               for j in range(len(weights)))
            for t in range(T)]


def _rmspe(gaps):
    if not gaps:
        return float("nan")
    return math.sqrt(sum(g * g for g in gaps) / len(gaps))


def _effect(gaps, t0, statistic, pre_gaps=None):
    post = gaps[t0:]
    if statistic == "effect":
        return sum(post) / len(post)
    pre = gaps[:t0] if pre_gaps is None else pre_gaps
    denom = _rmspe(pre)
    if denom <= 0:
        return float("inf")
    return _rmspe(post) / denom


def plcbsc(y_treated, y_donors, t0, x_treated=None, x_donors=None, v=None,
           statistic="effect", **fit_kwargs):
    r"""Synthetic control estimate with in-space placebo inference.

    Parameters
    ----------
    y_treated : sequence of float
        Outcome path of the treated unit, all periods.
    y_donors : sequence of sequences
        Outcome paths of the donor pool, one per donor, same length.
    t0 : int
        Index of the first post-intervention period.
    x_treated, x_donors : optional
        Predictors used to choose the weights. Omitted, the
        pre-intervention outcomes themselves are used, which is the common
        practice the paper warns can overfit when the donor pool is large.
    v : sequence of float, optional
        Diagonal of :math:`V`.
    statistic : {"effect", "rmspe_ratio"}
        What to permute. ``"effect"`` is the paper's own comparison, the
        mean post-intervention gap. ``"rmspe_ratio"`` is the post- over
        pre-intervention RMSPE, offered here as an option and NOT
        attributed to this paper.

    Returns
    -------
    RichResult
        ``estimate`` is the treated unit's statistic; ``gaps`` the
        per-period gaps; ``weights`` the synthetic control weights;
        ``placebo`` the donors' statistics; ``pvalue`` the fraction of
        placebo statistics at least as large in magnitude, including the
        treated unit itself; ``rank`` its rank; plus ``rmspe_pre``,
        ``rmspe_post`` and the fit diagnostics.

    Examples
    --------
    ::

        r = plcbsc(y1, donors, t0=10)
        r["estimate"], r["pvalue"]

    References
    ----------
    Abadie, Diamond & Hainmueller (2015) *AJPS* 59(2), 495-510: the
    simplex-constrained weights and the in-space placebo p-value.
    """
    y1 = [float(t) for t in y_treated]
    Y0 = [[float(t) for t in row] for row in y_donors]
    T = len(y1)
    J = len(Y0)
    if J == 0:
        raise ValueError("plcbsc: the donor pool is empty")
    if any(len(row) != T for row in Y0):
        raise ValueError("plcbsc: every donor needs the same number of "
                         "periods as the treated unit")
    t0 = int(t0)
    if not 1 <= t0 < T:
        raise ValueError("plcbsc: t0 must leave at least one pre- and one "
                         "post-intervention period")
    if statistic not in ("effect", "rmspe_ratio"):
        raise ValueError("plcbsc: statistic must be 'effect' or "
                         "'rmspe_ratio'")

    def predictors(unit_y, others_y):
        if x_treated is None:
            return unit_y[:t0], [o[:t0] for o in others_y]
        return None, None

    def fit(unit_y, others_y, unit_x, others_x):
        if unit_x is None:
            unit_x = unit_y[:t0]
            others_x = [o[:t0] for o in others_y]
        return synthetic_control(unit_x, others_x, v, **fit_kwargs)

    main = fit(y1, Y0, x_treated, x_donors)
    gaps = _gaps(y1, Y0, main["weights"])
    est = _effect(gaps, t0, statistic)

    placebo = []
    for j in range(J):
        others = [Y0[m] for m in range(J) if m != j]
        ox = None
        if x_donors is not None:
            ox = [x_donors[m] for m in range(J) if m != j]
        pf = fit(Y0[j], others, None if x_donors is None else x_donors[j],
                 ox)
        pg = _gaps(Y0[j], others, pf["weights"])
        placebo.append(_effect(pg, t0, statistic))

    all_stats = [abs(est)] + [abs(p) for p in placebo]
    at_least = sum(1 for s in all_stats if s >= abs(est) - 1e-12)
    pvalue = at_least / float(len(all_stats))
    rank = sorted(all_stats, reverse=True).index(abs(est)) + 1
    return RichResult(payload={
        "estimate": est,
        "gaps": gaps,
        "weights": main["weights"],
        "fit_loss": main["loss"],
        "placebo": placebo,
        "pvalue": pvalue,
        "rank": rank,
        "n_donors": J,
        "t0": t0,
        "statistic": statistic,
        "rmspe_pre": _rmspe(gaps[:t0]),
        "rmspe_post": _rmspe(gaps[t0:]),
        "note": "inference is by permutation over the donor pool, so the "
                "smallest attainable p-value is 1/(J+1) = %.4g"
                % (1.0 / (J + 1)),
        "method": "synthetic control with in-space placebos (Abadie, "
                  "Diamond & Hainmueller 2015)",
    })


def in_time_placebo(y_treated, y_donors, t0, fake_t0, v=None, **fit_kwargs):
    r"""The paper's in-time placebo: move the intervention earlier.

    Fits the synthetic control using only periods before ``fake_t0`` and
    reports the gaps between ``fake_t0`` and the real ``t0``. "Our
    confidence ... would dissipate if the synthetic control method also
    assigned a large effect to a period in which the intervention did not
    occur", so a large statistic here is bad news for the design.
    """
    y1 = [float(t) for t in y_treated]
    Y0 = [[float(t) for t in row] for row in y_donors]
    fake_t0 = int(fake_t0)
    t0 = int(t0)
    if not 1 <= fake_t0 < t0:
        raise ValueError("plcbsc: fake_t0 must fall before the real t0 and "
                         "leave a pre-period")
    fit = synthetic_control(y1[:fake_t0], [o[:fake_t0] for o in Y0], v,
                            **fit_kwargs)
    gaps = _gaps(y1, Y0, fit["weights"])
    return {"weights": fit["weights"], "gaps": gaps,
            "placebo_effect": sum(gaps[fake_t0:t0]) / (t0 - fake_t0),
            "rmspe_pre": _rmspe(gaps[:fake_t0]),
            "rmspe_placebo": _rmspe(gaps[fake_t0:t0])}


def cheatsheet():
    return ("plcbsc: synthetic control + placebo inference (Abadie, "
            "Diamond & Hainmueller 2015). Weights live on the SIMPLEX -- "
            "non-negative, summing to one -- which is what stops the "
            "counterfactual extrapolating outside the donors' support, "
            "unlike regression weights. No standard errors: run the whole "
            "procedure pretending each donor was treated, and the p-value "
            "is the fraction of placebo effects at least as large as the "
            "real one, so the smallest attainable p-value is 1/(J+1). "
            "in_time_placebo moves the date instead of the unit. The "
            "post/pre RMSPE ratio is offered as an option, not attributed "
            "to this paper.")


# compact alias per ledger/NAMING.md
placebo_inference = plcbsc

# name carried over from the generated stub this replaced
placebo_scm_inference = plcbsc
