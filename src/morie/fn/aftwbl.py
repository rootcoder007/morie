# morie.fn -- function file (rootcoder007/morie)
"""Weibull AFT model."""

from __future__ import annotations

import numpy as np

from ._aft import aft_fit
from ._richresult import RichResult
from ._surv import prepare

__all__ = ["aft_weibull"]


def aft_weibull(time, event, X, **kwargs):
    r"""Weibull AFT model.

    An accelerated-failure-time model is a linear model on **log time**,

    .. math::
        \log T_i = x_i^\top \beta + \sigma \epsilon_i,

    so :math:`e^{\beta_j}` is a **time ratio**: the multiplicative effect on
    survival time. A positive coefficient means *longer* survival. This is the
    opposite sign convention to Cox, where a positive coefficient raises the
    hazard and shortens survival, and confusing the two is the most common
    misreading of an AFT fit -- so ``time_ratio`` is returned explicitly rather
    than leaving ``exp(beta)`` to be interpreted by guesswork.

    The Weibull is the only distribution that is **both** an AFT model and a
    proportional-hazards model, which is why it is the usual first parametric
    choice and why its output can be read either way. The two parameterisations
    are linked by :math:`\beta_{\text{Cox}} = -\beta_{\text{AFT}}/\sigma`,
    and that identity is checked in the doctest below against a Cox fit on the
    same data.

    :math:`\sigma < 1` gives an increasing hazard (wear-out), :math:`\sigma > 1`
    a decreasing one (early failures), and :math:`\sigma = 1` the constant
    hazard of the exponential model.

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
    Weibull, W. (1951). A statistical distribution function of wide
        applicability. *Journal of Applied Mechanics*, 18, 293-297.
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis
        of Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    Coefficients are recovered on data simulated from the model.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(800, 2))
    >>> mu = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    >>> T = np.exp(mu + 0.6 * np.log(rng.exponential(1.0, 800)))
    >>> C = rng.exponential(float(np.exp(mu).mean()) * 4, 800)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = aft_weibull(t, e, X)
    >>> [bool(abs(r["beta"][i] - v) < 0.12) for i, v in enumerate([1.0, 0.7, -0.4])]
    [True, True, True]
    >>> bool(abs(r["sigma"] - 0.6) < 0.08)
    True

    The Weibull is simultaneously an AFT and a PH model, so its coefficients
    map onto a Cox fit by ``beta_cox = -beta_aft / sigma``.

    >>> from morie.fn.efrnt import efron_tie_correction
    >>> cox = efron_tie_correction(t, e, X)["beta"]
    >>> implied = -r["beta"][1:] / r["sigma"]
    >>> bool(np.max(np.abs(implied - cox)) < 0.15)
    True

    A positive coefficient means longer survival -- the opposite of Cox.

    >>> bool(r["time_ratio"][1] > 1.0 and cox[0] < 0)
    True
    """
    t, e, Xm = prepare(time, event, X)
    if np.any(t <= 0):
        raise ValueError("AFT models need strictly positive times")
    beta, log_sigma, ll, cov, it, conv = aft_fit(t, e, Xm, family="weibull", **kwargs)
    p = beta.size
    se = (np.sqrt(np.clip(np.diag(cov)[:p], 0, None))
          if cov is not None else np.full(p, np.nan))
    return RichResult(
        title="Weibull AFT model",
        summary_lines=[("n", int(t.size)), ("events", int(e.sum())),
                       ("sigma", float(np.exp(log_sigma))), ("loglik", ll)],
        warnings=[] if conv else ["the optimiser did not converge"],
        payload={
            "beta": beta, "se": se, "time_ratio": np.exp(beta),
            "sigma": float(np.exp(log_sigma)), "log_sigma": log_sigma,
            "loglik": ll, "aic": float(2 * (p + 1) - 2 * ll),
            "family": "weibull", "n": int(t.size), "n_events": int(e.sum()),
            "n_iter": it, "converged": conv, "cov": cov,
            "time": t, "event": e, "X": Xm,
            "method": "aft_weibull",
        },
    )


def cheatsheet():
    return "aftwbl: AFT -- exp(beta) is a TIME RATIO; positive beta means LONGER survival, unlike Cox"
