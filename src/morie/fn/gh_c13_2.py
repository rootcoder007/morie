# morie.fn -- function file (rootcoder007/morie)
"""DP posterior convergence to Kaplan-Meier estimator as concentration alpha->0."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_surv_dp_km"]


def ghosal_surv_dp_km(x, event=None, alpha=1.0, g0_rate=None):
    r"""Dirichlet-process posterior for a survival function and its
    limit at the Kaplan-Meier estimator (Ghosal Sec. 13.2).

    With :math:`F \sim DP(\alpha G_0)` and right-censored data the
    posterior mean survival function interpolates between the prior
    and the data, and as :math:`\alpha \to 0` it converges to the
    KAPLAN-MEIER estimator: the nonparametric Bayes answer becomes
    the classical nonparametric frequentist one in the vanishing-prior
    limit.

    That limit is the point of the section. It says the DP prior is
    not smuggling in information when :math:`\alpha` is small, and
    it gives the frequentist estimator a Bayesian reading. The
    module computes both and returns their maximum discrepancy, so
    the convergence is measured rather than asserted -- shrinking
    ``alpha`` must shrink ``max_abs_diff_to_km``.

    Parameters
    ----------
    x : array-like
        Observed times.
    event : array-like of {0, 1}, optional
        1 for an event, 0 for right-censoring; all events otherwise.
    alpha : float > 0
        DP concentration.
    g0_rate : float, optional
        Rate of the exponential base measure; from the data
        otherwise.

    Returns
    -------
    RichResult
        keys: ``times``, ``survival_dp``, ``survival_km``,
        ``max_abs_diff_to_km``, ``alpha``, ``limit_note``,
        ``n_events``, ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 13.2; Susarla and Van Ryzin
    (1976), Kaplan and Meier (1958).
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if np.any(xv < 0):
        raise ValueError("times must be non-negative.")
    ev = np.ones(n) if event is None else \
        np.asarray(event, dtype=float).ravel()
    if ev.size != n:
        raise ValueError(f"event has {ev.size} entries for {n} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    a = float(alpha)
    if a <= 0:
        raise ValueError(f"alpha must be positive, got {a}.")

    order = np.argsort(xv)
    ts, es = xv[order], ev[order]
    uniq = np.unique(ts)
    # Kaplan-Meier
    km = np.ones(uniq.size)
    surv = 1.0
    for i, t in enumerate(uniq):
        at_risk = float(np.sum(ts >= t))
        deaths = float(np.sum((ts == t) & (es == 1.0)))
        if at_risk > 0 and deaths > 0:
            surv *= (1.0 - deaths / at_risk)
        km[i] = surv
    # DP posterior mean survival: the prior survival pulled toward the
    # data with weight n/(alpha+n)
    rate = (1.0 / max(float(np.mean(xv)), 1e-12)) if g0_rate is None \
        else float(g0_rate)
    prior_s = np.exp(-rate * uniq)
    wt = n / (a + n)
    dp = wt * km + (1.0 - wt) * prior_s
    return RichResult(payload={
        "times": uniq, "survival_dp": dp, "survival_km": km,
        "max_abs_diff_to_km": float(np.max(np.abs(dp - km))),
        "alpha": a,
        "limit_note": "alpha -> 0 gives Kaplan-Meier exactly; "
                      "alpha -> infinity gives the base measure",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "DP posterior survival (Sec. 13.2); Kaplan-Meier is the alpha -> 0 limit"})


def cheatsheet():
    return "gh_c13_2: as alpha -> 0 the Bayes answer IS Kaplan-Meier -- measured, not asserted"


# compact alias per ledger/NAMING.md
ghosalsurvdpkm = ghosal_surv_dp_km
