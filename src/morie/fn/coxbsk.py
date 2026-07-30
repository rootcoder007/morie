# morie.fn -- function file (rootcoder007/morie)
"""Breslow baseline hazard for a fitted Cox model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from ._surv import baseline_hazard, cox_fit, prepare

__all__ = ["cox_breslow_step"]


def cox_breslow_step(time, event, X, beta=None, ties="efron"):
    r"""Breslow's step estimator of the baseline cumulative hazard.

    The Cox partial likelihood deliberately never estimates
    :math:`\lambda_0(t)` -- that is what "semiparametric" buys. To turn a
    fitted model into an actual *survival curve* the baseline must be put back,
    and Breslow's estimator does it as a step function jumping only at observed
    event times:

    .. math::
        \hat\Lambda_0(t) = \sum_{t_i \le t}
            \frac{d_i}{\sum_{k \in R_i} e^{\beta^\top x_k}},
        \qquad \hat S_0(t) = e^{-\hat\Lambda_0(t)} .

    Two consequences worth stating. The estimate is undefined beyond the last
    observed event time -- there is no risk set left, so the curve is flat
    there by convention rather than by evidence. And "baseline" means
    :math:`x = 0`, which for un-centred covariates is often a subject who
    could not exist; centring the covariates first makes the baseline the
    average subject and the curve interpretable.

    Parameters
    ----------
    time, event, X : array-like
        Survival data as elsewhere in this family.
    beta : array-like, optional
        Coefficients. Fitted internally when omitted.
    ties : {"efron", "breslow"}
        Tie handling for the internal fit.

    Returns
    -------
    RichResult
        ``times``, ``hazard`` (increments), ``cumhazard``, ``survival``,
        ``beta``.

    References
    ----------
    Breslow, N. (1972). Discussion of Professor Cox's paper. *JRSS-B*, 34,
        216-217.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    The baseline cumulative hazard is non-decreasing and survival is
    non-increasing -- the defining shape constraints.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 1))
    >>> T = rng.exponential(1 / np.exp(X[:, 0] * 0.7))
    >>> C = rng.exponential(2.0, 300)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_breslow_step(t, e, X)
    >>> bool(np.all(np.diff(r["cumhazard"]) >= 0))
    True
    >>> bool(np.all(np.diff(r["survival"]) <= 0))
    True

    Survival starts near 1 and stays in the unit interval.

    >>> bool(r["survival"][0] <= 1.0 and r["survival"].min() >= 0.0)
    True

    Jumps occur only at observed event times.

    >>> bool(np.array_equal(r["times"], np.unique(t[e == 1])))
    True
    """
    t, e, Xm = prepare(time, event, X)
    if beta is None:
        beta, *_ = cox_fit(t, e, Xm, ties=ties)
    beta = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    if beta.size != Xm.shape[1]:
        raise ValueError(f"beta has {beta.size} entries but X has {Xm.shape[1]} columns")
    times, dH, H = baseline_hazard(t, e, Xm, beta)
    return RichResult(
        title="Breslow baseline hazard",
        summary_lines=[("event times", int(times.size)),
                       ("max cumhazard", float(H[-1]) if H.size else float("nan"))],
        warnings=["the baseline is undefined beyond the last event time; the "
                  "curve is flat there by convention, not by evidence"],
        payload={
            "times": times, "hazard": dH, "cumhazard": H,
            "survival": np.exp(-H), "beta": beta,
            "n": int(t.size), "method": "cox_breslow_step",
        },
    )


def cheatsheet():
    return "coxbsk: puts lambda_0 back; 'baseline' means x=0, so CENTRE covariates or it is meaningless"
