# morie.fn -- function file (rootcoder007/morie)
r"""Time-dependent ROC for censored survival data.

**Why the ordinary ROC does not apply.** A diagnostic marker for a
survival outcome has no fixed case/control split: whether a subject
is a case depends on the horizon you ask about. Heagerty, Lumley and
Pepe define the *cumulative case / dynamic control* classification at
time :math:`t` --

.. math:: \mathrm{se}(c, t) = P(M > c \mid T \le t), \qquad
          \mathrm{sp}(c, t) = P(M \le c \mid T > t)

-- so every subject is a control until it fails and a case thereafter,
and the ROC curve and its area are functions of :math:`t`.

**Censoring is the whole difficulty.** A subject censored before
:math:`t` has unknown case status, so the empirical proportions above
are not computable. Dropping such subjects biases the result whenever
censoring is related to the marker. The paper's estimator instead
routes through Bayes' theorem and the Kaplan-Meier estimator:

.. math:: \widehat{\mathrm{se}}(c, t) =
          \frac{\{1 - \hat{S}_{M > c}(t)\}\, \hat{P}(M > c)}
               {1 - \hat{S}(t)}, \qquad
          \widehat{\mathrm{sp}}(c, t) =
          \frac{\hat{S}_{M \le c}(t)\, \hat{P}(M \le c)}
               {\hat{S}(t)}

with :math:`\hat{S}_{M>c}` the Kaplan-Meier curve computed within the
subset above the threshold. Nothing is discarded: a censored subject
contributes to the risk sets of whichever subset it belongs to.

**The check that matters.** With no censoring, the estimator must
collapse to the plain empirical proportions, and the area under the
curve must equal the Mann-Whitney statistic comparing markers of
cases against controls. Both are exact identities and both are
anchored -- an estimator that merely looks plausible under censoring
but misses these is wrong.

**Two routes.** ``km`` is the estimator above. ``empirical`` computes
the proportions directly and is *only* valid without censoring; it is
provided because it is the thing the KM route must reduce to, and it
refuses to run on censored data rather than quietly discarding
subjects.

**A caveat the estimator carries.** The KM route can produce
sensitivities outside :math:`[0, 1]` when the marker-specific curves
cross badly at small samples. That is a property of the estimator,
not a bug; the value is reported along with a flag rather than
clipped silently.

References
----------
Heagerty, P. J., Lumley, T. & Pepe, M. S. (2000) "Time-dependent ROC
curves for censored survival data and a diagnostic marker",
*Biometrics* 56(2), 337-344,
doi:10.1111/j.0006-341X.2000.00337.x. Sec. 2 (the cumulative
case / dynamic control definitions above), Sec. 2.1 (the
Kaplan-Meier estimator of sensitivity and specificity reproduced
here, and its reduction to the empirical estimator without
censoring), and the time-dependent area under the curve.
"""

import math

from ._richresult import RichResult

__all__ = ["kaplan_meier", "roc_at", "auc_at", "sensitivity",
           "specificity", "ROUTES", "time_dependent_roc"]

ROUTES = ("km", "empirical")


def _clean(times, events, marker=None):
    T = [float(x) for x in times]
    E = [int(x) for x in events]
    if len(T) != len(E):
        raise ValueError("survroc: %d times but %d event indicators"
                         % (len(T), len(E)))
    if not T:
        raise ValueError("survroc: no subjects given")
    if any(x < 0 for x in T):
        raise ValueError("survroc: a survival time cannot be "
                         "negative")
    if any(e not in (0, 1) for e in E):
        raise ValueError("survroc: the event indicator must be 0 "
                         "(censored) or 1 (event)")
    if marker is None:
        return T, E, None
    M = [float(x) for x in marker]
    if len(M) != len(T):
        raise ValueError("survroc: %d markers but %d subjects"
                         % (len(M), len(T)))
    return T, E, M


def kaplan_meier(times, events, at=None):
    r"""The product-limit estimator of :math:`S(t)`.

    With ``at`` given, the survival probability at that time; without
    it, the step function as (time, survival) pairs.
    """
    T, E, _ = _clean(times, events)
    n = len(T)
    order = sorted(range(n), key=lambda i: (T[i], -E[i]))
    curve, s, at_risk, i = [(0.0, 1.0)], 1.0, n, 0
    while i < n:
        t = T[order[i]]
        d = k = 0
        while i < n and T[order[i]] == t:
            d += E[order[i]]
            k += 1
            i += 1
        if d:
            s *= 1.0 - d / float(at_risk)
            curve.append((t, s))
        at_risk -= k
    if at is None:
        return curve
    val = 1.0
    for t, s in curve:
        if t <= float(at):
            val = s
        else:
            break
    return val


def _empirical(T, E, M, c, t):
    if any(e == 0 and s < t for s, e in zip(T, E)):
        raise ValueError("survroc: the empirical route needs complete "
                         "follow-up to time %g, but a subject is "
                         "censored before it" % t)
    case = [i for i in range(len(T)) if T[i] <= t and E[i] == 1]
    ctrl = [i for i in range(len(T)) if T[i] > t]
    if not case or not ctrl:
        raise ValueError("survroc: at t = %g there are %d cases and "
                         "%d controls; both are needed"
                         % (t, len(case), len(ctrl)))
    se = sum(1 for i in case if M[i] > c) / float(len(case))
    sp = sum(1 for i in ctrl if M[i] <= c) / float(len(ctrl))
    return se, sp


def _km_pair(T, E, M, c, t):
    n = len(T)
    hi = [i for i in range(n) if M[i] > c]
    lo = [i for i in range(n) if M[i] <= c]
    S = kaplan_meier(T, E, t)
    if S <= 0.0:
        raise ValueError("survroc: the overall survival estimate is "
                         "zero at t = %g, so specificity is not "
                         "defined there" % t)
    if S >= 1.0:
        raise ValueError("survroc: no events by t = %g, so "
                         "sensitivity is not defined there" % t)
    p_hi = len(hi) / float(n)
    p_lo = len(lo) / float(n)
    s_hi = (kaplan_meier([T[i] for i in hi], [E[i] for i in hi], t)
            if hi else 1.0)
    s_lo = (kaplan_meier([T[i] for i in lo], [E[i] for i in lo], t)
            if lo else 1.0)
    se = (1.0 - s_hi) * p_hi / (1.0 - S)
    sp = s_lo * p_lo / S
    return se, sp


def sensitivity(times, events, marker, threshold, t, route="km"):
    r"""P(M > c | T <= t) at one threshold."""
    return _pair(times, events, marker, threshold, t, route)[0]


def specificity(times, events, marker, threshold, t, route="km"):
    r"""P(M <= c | T > t) at one threshold."""
    return _pair(times, events, marker, threshold, t, route)[1]


def _pair(times, events, marker, c, t, route):
    if route not in ROUTES:
        raise ValueError("survroc: route must be one of %s, got %r"
                         % (", ".join(ROUTES), route))
    T, E, M = _clean(times, events, marker)
    tt = float(t)
    if tt <= 0.0:
        raise ValueError("survroc: the horizon must be positive")
    if route == "empirical":
        return _empirical(T, E, M, float(c), tt)
    return _km_pair(T, E, M, float(c), tt)


def roc_at(times, events, marker, t, route="km"):
    r"""The whole curve at horizon ``t``, ordered by threshold.

    Thresholds are taken just below each distinct marker value, plus
    the two endpoints, so the curve runs from (1, 0) to (0, 1).
    """
    T, E, M = _clean(times, events, marker)
    vals = sorted(set(M))
    eps = (max(vals) - min(vals)) or 1.0
    cuts = ([min(vals) - eps] + [v for v in vals]
            + [max(vals) + eps])
    pts = []
    for c in cuts:
        se, sp = _pair(times, events, marker, c, t, route)
        pts.append({"threshold": c, "sensitivity": se,
                    "specificity": sp, "fpr": 1.0 - sp})
    # Sort by threshold, descending. Sensitivity is non-increasing
    # and specificity non-decreasing in the threshold, so this puts
    # the curve in order exactly -- sorting on the computed false
    # positive rate instead shuffles vertices whose rate should be
    # identical but differs in the last bits of the estimator.
    pts.sort(key=lambda p: -p["threshold"])
    return pts


def auc_at(times, events, marker, t, route="km"):
    r"""Area under the time-dependent ROC, by the trapezoid rule."""
    pts = roc_at(times, events, marker, t, route)
    a = 0.0
    for p, q in zip(pts, pts[1:]):
        a += (q["fpr"] - p["fpr"]) * (p["sensitivity"]
                                      + q["sensitivity"]) / 2.0
    return a


def time_dependent_roc(times, events, marker, t, route="km"):
    r"""Entry point: the curve, its area, and the case/control counts."""
    T, E, M = _clean(times, events, marker)
    pts = roc_at(times, events, marker, t, route)
    a = 0.0
    for p, q in zip(pts, pts[1:]):
        a += (q["fpr"] - p["fpr"]) * (p["sensitivity"]
                                      + q["sensitivity"]) / 2.0
    out_of_range = [p for p in pts
                    if not (-1e-9 <= p["sensitivity"] <= 1 + 1e-9
                            and -1e-9 <= p["specificity"] <= 1 + 1e-9)]
    return RichResult(payload={
        "estimate": a, "auc": a, "roc": pts, "horizon": float(t),
        "route": route,
        "n": len(T), "n_events_by_t": sum(1 for i in range(len(T))
                                          if T[i] <= t and E[i] == 1),
        "n_at_risk_after_t": sum(1 for x in T if x > t),
        "n_censored_before_t": sum(1 for i in range(len(T))
                                   if T[i] < t and E[i] == 0),
        "survival_at_t": kaplan_meier(T, E, t),
        "out_of_range": out_of_range,
        "method": "Heagerty, Lumley & Pepe (2000) cumulative "
                  "case / dynamic control ROC, %s estimator" % route,
    })
