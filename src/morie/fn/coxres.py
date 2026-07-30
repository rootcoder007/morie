# morie.fn -- function file (rootcoder007/morie)
"""Schoenfeld residuals and the proportional-hazards test."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["cox_schoenfeld_residuals"]


def cox_schoenfeld_residuals(fit, transform="km"):
    r"""Schoenfeld residuals, one per event, and the test they support.

    At each event time the residual is the covariate of the subject who failed
    minus the risk-set average weighted by the fitted hazard:

    .. math::
        s_i = x_{(i)} - \frac{\sum_{k \in R_i} e^{\beta^\top x_k} x_k}
                             {\sum_{k \in R_i} e^{\beta^\top x_k}} .

    Censored subjects contribute nothing, so there are as many residuals as
    events, not as subjects.

    Their diagnostic value is entirely about **proportional hazards**. Under
    PH the residuals have zero mean and no trend in time; a slope against
    (transformed) time is direct evidence that the hazard ratio is changing --
    that is the Grambsch-Therneau test, and its correlation statistic is
    returned here.

    A significant result does not mean the covariate is unimportant. It means
    a *single* hazard ratio is the wrong summary, and the fix is stratification
    or a time-varying coefficient, not deletion.

    Parameters
    ----------
    fit : mapping
        A result from one of the Cox fitters.
    transform : {"km", "rank", "identity"}
        Time transform for the trend test. ``"km"`` (the default in most
        software) weights by the Kaplan-Meier estimate.

    Returns
    -------
    RichResult
        ``residuals`` ``(n_events, p)``, ``times``, ``correlation``,
        ``p_value``, ``transform``.

    References
    ----------
    Schoenfeld, D. (1982). Partial residuals for the proportional hazards
        regression model. *Biometrika*, 69(1), 239-241.
    Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and
        diagnostics based on weighted residuals. *Biometrika*, 81(3), 515-526.

    Examples
    --------
    Under proportional hazards there is no trend, so the test does not fire.

    >>> import numpy as np
    >>> from morie.fn.efrnt import efron_tie_correction
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 1))
    >>> T = rng.exponential(1 / np.exp(X[:, 0] * 0.9))
    >>> C = rng.exponential(3.0, 400)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_schoenfeld_residuals(efron_tie_correction(t, e, X))
    >>> bool(r["p_value"][0] > 0.05)
    True

    One residual per event, not per subject.

    >>> int(r["residuals"].shape[0]) == int(e.sum())
    True

    Residuals have approximately zero mean under a correct fit.

    >>> bool(abs(float(r["residuals"].mean())) < 0.1)
    True
    """
    from scipy.stats import norm

    t = np.asarray(fit["time"], dtype=float)
    e = np.asarray(fit["event"], dtype=float)
    X = np.atleast_2d(np.asarray(fit["X"], dtype=float))
    beta = np.asarray(fit["beta"], dtype=float).ravel()
    w = np.exp(np.clip(X @ beta, -500, 500))

    ev_idx = np.flatnonzero(e == 1)
    ev_idx = ev_idx[np.argsort(t[ev_idx])]
    res = np.empty((ev_idx.size, X.shape[1]))
    for r_i, i in enumerate(ev_idx):
        at_risk = t >= t[i]
        wr = w[at_risk]
        res[r_i] = X[i] - (wr @ X[at_risk]) / wr.sum()
    times = t[ev_idx]

    if transform == "km":
        from ._surv import km_estimate

        ut, surv = km_estimate(t, e)
        idx = np.clip(np.searchsorted(ut, times, side="right") - 1, 0, surv.size - 1)
        g = 1.0 - surv[idx]
    elif transform == "rank":
        g = np.argsort(np.argsort(times)).astype(float)
    elif transform == "identity":
        g = times
    else:
        raise ValueError('transform must be "km", "rank" or "identity"')

    p = X.shape[1]
    corr = np.full(p, np.nan)
    pval = np.full(p, np.nan)
    if res.shape[0] > 2 and np.ptp(g) > 0:
        for j in range(p):
            if np.std(res[:, j]) == 0:
                continue
            c = float(np.corrcoef(g, res[:, j])[0, 1])
            corr[j] = c
            zst = c * np.sqrt(max(res.shape[0] - 2, 1) / max(1 - c**2, 1e-12))
            pval[j] = float(2 * norm.sf(abs(zst)))
    return RichResult(
        title="Schoenfeld residuals",
        summary_lines=[("events", int(res.shape[0])), ("transform", transform)],
        payload={
            "residuals": res, "times": times, "correlation": corr,
            "p_value": pval, "transform": transform,
            "method": "cox_schoenfeld_residuals",
        },
    )


def cheatsheet():
    return "coxres: one per EVENT; a time trend means PH fails -- stratify, do not drop the covariate"
