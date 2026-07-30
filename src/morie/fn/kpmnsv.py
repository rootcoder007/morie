# morie.fn -- function file (rootcoder007/morie)
"""Kaplan-Meier survival estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kaplan_meier", "kaplan_meier_survival"]


def kaplan_meier(time, event, alpha=0.05, conf_type="log-log"):
    r"""Product-limit estimator with Greenwood variance.

    .. math::
       \hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right),
       \qquad
       \widehat{\mathrm{Var}}[\hat S(t)] = \hat S(t)^2
       \sum_{t_i\le t}\frac{d_i}{n_i(n_i-d_i)}

    Censored observations leave the risk set without contributing a
    drop, which is exactly how the estimator uses partial information
    instead of discarding it. The assumption that buys this is
    INDEPENDENT censoring: those censored at :math:`t` must have the
    same future risk as those still under observation. When censoring
    is informative -- patients withdrawing because they are
    deteriorating -- Kaplan-Meier is biased, and nothing in the data
    reveals it.

    The interval is on the LOG-LOG scale by default, not the linear
    one. A linear interval can run below 0 or above 1, which is not
    merely inelegant: near the tails, where the estimate is least
    precise, it is also where the linear interval is most likely to be
    nonsensical. The log-log transform keeps it inside :math:`[0,1]`
    by construction.

    ``tail_reliable`` flags where the risk set has fallen below 10, the
    region where the curve is driven by a handful of subjects and
    should not be read as an estimate.

    Parameters
    ----------
    time : array-like, shape (n,)
    event : array-like of {0, 1}, shape (n,)
        1 = event observed, 0 = right-censored.
    alpha : float
    conf_type : {'log-log', 'plain'}

    Returns
    -------
    RichResult
        ``times``, ``survival``, ``se``, ``ci_lower``, ``ci_upper``,
        ``at_risk``, ``events``, ``median``, ``rmst``,
        ``tail_reliable``.

    References
    ----------
    Kaplan and Meier (1958), *JASA* 53:457-481.
    Greenwood (1926) for the variance.
    Kalbfleisch and Prentice (2002), section 1.4, for the log-log
    interval.

    Examples
    --------
    >>> out = kaplan_meier([1, 2, 3], [1, 1, 1])
    >>> [round(float(s), 4) for s in out["survival"]]
    [0.6667, 0.3333, 0.0]
    """
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event, dtype=float).ravel()
    n = t.size
    if e.size != n:
        raise ValueError("time and event must agree in length.")
    if n == 0:
        raise ValueError("need at least one observation.")
    if not np.all(np.isin(e, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if np.any(t < 0):
        raise ValueError("time must be non-negative.")
    if conf_type not in ("log-log", "plain"):
        raise ValueError(
            "conf_type must be 'log-log' or 'plain', got %r." % conf_type
        )

    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq = np.unique(t[e == 1])
    S, var_sum = 1.0, 0.0
    times, surv, ses, risk, evs = [], [], [], [], []
    for u in uniq:
        nr = int(np.sum(t >= u))
        di = int(np.sum((t == u) & (e == 1)))
        if nr <= 0:
            continue
        S *= (1.0 - di / nr)
        if nr > di:
            var_sum += di / (nr * (nr - di))
        else:
            var_sum = np.inf
        times.append(float(u))
        surv.append(float(S))
        ses.append(float(S * np.sqrt(var_sum)) if np.isfinite(var_sum)
                   else np.nan)
        risk.append(nr)
        evs.append(di)
    times = np.asarray(times)
    surv = np.asarray(surv)
    ses = np.asarray(ses)
    risk = np.asarray(risk)
    evs = np.asarray(evs)

    z = 1.959963984540054
    if conf_type == "plain":
        lo = np.clip(surv - z * ses, 0.0, 1.0)
        hi = np.clip(surv + z * ses, 0.0, 1.0)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ls = np.log(np.clip(surv, 1e-300, 1.0))
            se_ll = np.where(surv > 0, ses / (surv * np.abs(ls)), np.nan)
            lo = surv ** np.exp(z * se_ll)
            hi = surv ** np.exp(-z * se_ll)
        lo = np.clip(np.nan_to_num(lo, nan=0.0), 0.0, 1.0)
        hi = np.clip(np.nan_to_num(hi, nan=1.0), 0.0, 1.0)

    med = np.nan
    below = np.nonzero(surv <= 0.5)[0]
    if below.size:
        med = float(times[below[0]])
    # restricted mean survival time by the step function up to the last time
    if times.size:
        edges = np.concatenate([[0.0], times])
        heights = np.concatenate([[1.0], surv])[:-1]
        rmst = float(np.sum(np.diff(edges) * heights))
    else:
        rmst = 0.0
    return RichResult(
        payload={
            "estimate": surv,
            "times": times,
            "survival": surv,
            "se": ses,
            "ci_lower": lo,
            "ci_upper": hi,
            "conf_type": conf_type,
            "ci_note": (
                "log-log intervals stay inside [0, 1] by construction; a "
                "linear interval can leave it exactly in the tails, where "
                "the estimate is least precise"
            ),
            "at_risk": risk,
            "events": evs,
            "median": med,
            "median_note": (
                None if med == med else
                "survival never reaches 0.5, so the median is not estimable "
                "from this follow-up"
            ),
            "rmst": rmst,
            "tail_reliable": (risk >= 10),
            "tail_note": (
                "where the risk set falls below 10 the curve is driven by a "
                "few subjects and should not be read as an estimate"
            ),
            "censoring_note": (
                "validity needs INDEPENDENT censoring: those censored at t "
                "must carry the same future risk as those still observed. "
                "Informative censoring biases the curve and leaves no trace "
                "in the data"
            ),
            "n_events": int(e.sum()),
            "n_censored": int((1 - e).sum()),
            "n": int(n),
            "method": "Kaplan-Meier product-limit estimator",
        }
    )


def cheatsheet():
    return (
        "kpmnsv: Kaplan-Meier with Greenwood variance, log-log intervals "
        "and a risk-set reliability flag"
    )


#: Catalogue alias for :func:`kaplan_meier`.
kaplan_meier_survival = kaplan_meier
