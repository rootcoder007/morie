# morie.fn -- function file (rootcoder007/morie)
"""Log-logistic AFT model."""

from __future__ import annotations

import numpy as np

from ._aft import aft_fit
from ._richresult import RichResult
from ._surv import prepare

__all__ = ["aft_log_logistic"]


def aft_log_logistic(time, event, X, **kwargs):
    r"""Log-logistic AFT model.

    An accelerated-failure-time model is a linear model on **log time**,

    .. math::
        \log T_i = x_i^\top \beta + \sigma \epsilon_i,

    so :math:`e^{\beta_j}` is a **time ratio**: the multiplicative effect on
    survival time. A positive coefficient means *longer* survival. This is the
    opposite sign convention to Cox, where a positive coefficient raises the
    hazard and shortens survival, and confusing the two is the most common
    misreading of an AFT fit -- so ``time_ratio`` is returned explicitly rather
    than leaving ``exp(beta)`` to be interpreted by guesswork.

    Logistic errors give a hazard that rises and then falls, which is the
    reason to choose this family: no proportional-hazards model can represent a
    hazard that peaks and declines, and neither can the Weibull, whose hazard
    is monotone by construction. Recurrence data with an early peak in risk is
    the canonical case.

    The log-logistic is an AFT model but **not** a proportional-hazards model,
    so its coefficients have no hazard-ratio reading at all. It does have a
    proportional-*odds* reading: :math:`e^{-\beta/\sigma}` is the odds ratio
    of surviving past any given time.

    Right-censored observations contribute the log survivor function rather
    than the log density, which is what makes the fit valid under censoring at
    all.

    Parameters
    ----------
    time : array-like
        Observed follow-up times, positive.
    event : array-like
        1 for the event, 0 for right-censoring.
    X : array-like
        Covariates ``(n, p)``. An intercept is added.
    **kwargs
        Passed to the optimiser.

    Returns
    -------
    RichResult
        ``beta`` (log-time scale, intercept first), ``time_ratio``, ``sigma``,
        ``loglik``, ``aic``, ``converged``.

    References
    ----------
    Bennett, S. (1983). Log-logistic regression models for survival data.
        *Applied Statistics*, 32(2), 165-171.
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis
        of Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(800, 2))
    >>> mu = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    >>> u = rng.random(800)
    >>> T = np.exp(mu + 0.6 * np.log(u / (1 - u)))
    >>> C = rng.exponential(float(np.exp(mu).mean()) * 4, 800)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = aft_log_logistic(t, e, X)
    >>> [bool(abs(r["beta"][i] - v) < 0.15) for i, v in enumerate([1.0, 0.7, -0.4])]
    [True, True, True]

    It fits log-logistic data better than a Weibull does, which is the whole
    reason to have the family available.

    >>> from morie.fn.aftwbl import aft_weibull
    >>> bool(r["aic"] < aft_weibull(t, e, X)["aic"])
    True
    """
    t, e, Xm = prepare(time, event, X)
    if np.any(t <= 0):
        raise ValueError("AFT models need strictly positive times")
    beta, log_sigma, ll, cov, it, conv = aft_fit(t, e, Xm, family="loglogistic", **kwargs)
    p = beta.size
    se = (np.sqrt(np.clip(np.diag(cov)[:p], 0, None))
          if cov is not None else np.full(p, np.nan))
    return RichResult(
        title="Log-logistic AFT model",
        summary_lines=[("n", int(t.size)), ("events", int(e.sum())),
                       ("sigma", float(np.exp(log_sigma))), ("loglik", ll)],
        warnings=[] if conv else ["the optimiser did not converge"],
        payload={
            "beta": beta, "se": se, "time_ratio": np.exp(beta),
            "sigma": float(np.exp(log_sigma)), "log_sigma": log_sigma,
            "loglik": ll, "aic": float(2 * (p + 1) - 2 * ll),
            "family": "loglogistic", "n": int(t.size), "n_events": int(e.sum()),
            "n_iter": it, "converged": conv, "cov": cov,
            "time": t, "event": e, "X": Xm,
            "method": "aft_log_logistic",
        },
    )


def cheatsheet():
    return "aftllg: AFT -- exp(beta) is a TIME RATIO; positive beta means LONGER survival, unlike Cox"
