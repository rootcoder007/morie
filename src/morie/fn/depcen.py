# morie.fn -- function file (rootcoder007/morie)
"""Dependent-censoring diagnostic for a Cox model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["dependent_censoring_hazard"]


def dependent_censoring_hazard(time, event, X, ties="efron"):
    r"""Test whether censoring depends on the covariates.

    Fits the Cox model twice: once for the event and once treating
    **censoring** as the outcome. Under independent censoring the censoring
    model's coefficients should be null -- censoring times carry no covariate
    information.

    Every standard survival estimator assumes censoring is independent of the
    event time given covariates. When it is not -- sicker patients withdrawing
    earlier -- Kaplan-Meier and Cox are both biased, and the bias has no
    general sign: informative censoring can inflate or deflate survival
    depending on whether the withdrawers were high or low risk.

    Nothing here proves independence. A covariate predicting censoring is
    handled by conditioning on it, and the assumption then only needs to hold
    *given* the model. What cannot be detected at all is censoring depending on
    the unobserved event time itself, which is why this is a diagnostic rather
    than a test of the assumption. The remedy when it fires is IPCW
    (:func:`~morie.fn.chrwgt.censoring_at_risk_weight`) or a sensitivity
    analysis, not reassurance.

    Parameters
    ----------
    time, event, X : array-like
        Survival data; ``event`` is 1 for the event, 0 for censoring.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``beta_censoring``, ``se``, ``p_value``, ``beta_event``,
        ``dependent``, ``n_flagged``.

    References
    ----------
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of
        Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    Independent censoring leaves the censoring model null.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1000, 1))
    >>> T = rng.exponential(1 / np.exp(0.8 * X[:, 0]))
    >>> C = rng.exponential(1.5, 1000)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> bool(not dependent_censoring_hazard(t, e, X)["dependent"])
    True

    Censoring driven by the same covariate is detected.

    >>> C2 = rng.exponential(1 / np.exp(0.9 * X[:, 0]))
    >>> t2, e2 = np.minimum(T, C2), (T <= C2).astype(float)
    >>> bool(dependent_censoring_hazard(t2, e2, X)["dependent"])
    True

    The event model is returned too, so the two can be compared.

    >>> r = dependent_censoring_hazard(t2, e2, X)
    >>> bool(abs(r["beta_event"][0]) > 0.3)
    True
    """
    t, e, Xm = prepare(time, event, X)
    from scipy.stats import norm

    b_ev, *_ = cox_fit(t, e, Xm, ties=ties)
    cen = 1.0 - e
    if cen.sum() == 0:
        raise ValueError("there are no censored observations to model")
    b_c, ll, I, _, _, conv = cox_fit(t, cen, Xm, ties=ties)
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(I)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(b_c.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = b_c / se
    p = 2 * norm.sf(np.abs(z))
    flagged = int(np.sum(p < 0.05))
    return RichResult(
        title="Dependent-censoring diagnostic",
        summary_lines=[("n", int(t.size)), ("censored", int(cen.sum())),
                       ("covariates predicting censoring", flagged)],
        warnings=["censoring that depends on the unobserved event time is "
                  "undetectable by any diagnostic; a null result here does not "
                  "establish independent censoring"],
        payload={
            "beta_censoring": b_c, "se": se, "z": z, "p_value": p,
            "beta_event": b_ev, "dependent": bool(flagged > 0),
            "n_flagged": flagged, "n_censored": int(cen.sum()),
            "converged": conv, "method": "dependent_censoring_hazard",
        },
    )


def cheatsheet():
    return "depcen: models CENSORING as the outcome; cannot see dependence on the unobserved event time"
