# morie.fn -- function file (rootcoder007/morie)
"""Log-normal AFT model."""

from __future__ import annotations

from . import _array_core as np

from ._aft import aft_fit
from ._richresult import RichResult
from ._surv import prepare

__all__ = ["aft_generalized_gamma"]


def aft_generalized_gamma(time, event, X, **kwargs):
    r"""Log-normal AFT model.

    An accelerated-failure-time model is a linear model on **log time**,

    .. math::
        \log T_i = x_i^\top \beta + \sigma \epsilon_i,

    so :math:`e^{\beta_j}` is a **time ratio**: the multiplicative effect on
    survival time. A positive coefficient means *longer* survival. This is the
    opposite sign convention to Cox, where a positive coefficient raises the
    hazard and shortens survival, and confusing the two is the most common
    misreading of an AFT fit -- so ``time_ratio`` is returned explicitly rather
    than leaving ``exp(beta)`` to be interpreted by guesswork.

    Normal errors on the log scale, so survival times are log-normal. The
    hazard rises to a peak and then falls toward zero -- like the log-logistic
    but with a lighter tail, which matters when the question is about long-term
    survivors.

    This is the family the generalised gamma nests along with the Weibull and
    the exponential, so it is the natural comparator when choosing among them
    by likelihood. It is an AFT model and neither proportional hazards nor
    proportional odds.

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
    Lawless, J. F. (2003). *Statistical Models and Methods for Lifetime
        Data* (2nd ed.). Wiley.
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis
        of Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> X = rng.normal(size=(800, 2))
    >>> mu = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    >>> T = np.exp(mu + 0.6 * rng.normal(size=800))
    >>> C = rng.exponential(float(np.exp(mu).mean()) * 4, 800)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = aft_generalized_gamma(t, e, X)
    >>> [bool(abs(r["beta"][i] - v) < 0.15) for i, v in enumerate([1.0, 0.7, -0.4])]
    [True, True, True]
    >>> bool(abs(r["sigma"] - 0.6) < 0.08)
    True

    Censored times must still be positive -- a log-time model has nothing to
    say about zero.

    >>> aft_generalized_gamma([0.0, 1.0], [1.0, 1.0], [[1.0], [2.0]])
    Traceback (most recent call last):
        ...
    ValueError: AFT models need strictly positive times
    """
    t, e, Xm = prepare(time, event, X)
    if np.any(t <= 0):
        raise ValueError("AFT models need strictly positive times")
    beta, log_sigma, ll, cov, it, conv = aft_fit(t, e, Xm, family="lognormal", **kwargs)
    p = beta.size
    se = (np.sqrt(np.clip(np.diag(cov)[:p], 0, None))
          if cov is not None else np.full(p, np.nan))
    return RichResult(
        title="Log-normal AFT model",
        summary_lines=[("n", int(t.size)), ("events", int(e.sum())),
                       ("sigma", float(np.exp(log_sigma))), ("loglik", ll)],
        warnings=[] if conv else ["the optimiser did not converge"],
        payload={
            "beta": beta, "se": se, "time_ratio": np.exp(beta),
            "sigma": float(np.exp(log_sigma)), "log_sigma": log_sigma,
            "loglik": ll, "aic": float(2 * (p + 1) - 2 * ll),
            "family": "lognormal", "n": int(t.size), "n_events": int(e.sum()),
            "n_iter": it, "converged": conv, "cov": cov,
            "time": t, "event": e, "X": Xm,
            "method": "aft_generalized_gamma",
        },
    )


def cheatsheet():
    return "aftgma: AFT -- exp(beta) is a TIME RATIO; positive beta means LONGER survival, unlike Cox"
