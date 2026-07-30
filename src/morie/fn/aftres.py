# morie.fn -- function file (rootcoder007/morie)
"""Residuals for an accelerated-failure-time model."""

from __future__ import annotations

import numpy as np

from ._aft import log_dens_surv
from ._richresult import RichResult

__all__ = ["aft_residuals"]


def aft_residuals(fit):
    r"""Standardised, Cox-Snell and martingale residuals for an AFT fit.

    Three residuals, each answering a different question.

    **Standardised** :math:`z_i = (\log t_i - x_i^\top\beta)/\sigma` are the
    fitted errors. Under a correct model the uncensored ones follow the
    family's assumed error distribution -- extreme-value, logistic or normal --
    so a Q-Q plot of them against that distribution checks the family choice
    directly.

    **Cox-Snell** :math:`r_i = -\log \hat S(t_i \mid x_i)` are the key
    diagnostic: under a correct model they are a censored sample from a
    **unit exponential**, whatever the family. Their Nelson-Aalen cumulative
    hazard should therefore lie on the 45-degree line, which is a single plot
    that checks the whole fit rather than one assumption.

    **Martingale** :math:`\delta_i - r_i` are used as in the Cox case, for
    functional form of an omitted covariate.

    The unit-exponential property is asserted in the doctest rather than
    described, because it is the one that makes the diagnostic worth anything.

    Parameters
    ----------
    fit : mapping
        A result from :func:`~morie.fn.aftwbl.aft_weibull`,
        :func:`~morie.fn.aftllg.aft_log_logistic` or
        :func:`~morie.fn.aftgma.aft_generalized_gamma`.

    Returns
    -------
    RichResult
        ``standardized``, ``cox_snell``, ``martingale``, ``deviance``,
        ``family``.

    References
    ----------
    Cox, D. R., & Snell, E. J. (1968). A general definition of residuals.
        *JRSS-B*, 30(2), 248-265.
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis
        of Failure Time Data* (2nd ed.). Wiley.

    Examples
    --------
    Cox-Snell residuals from a correct fit are a unit-exponential sample: mean
    about 1 among the uncensored, which is the property the diagnostic rests
    on.

    >>> import numpy as np
    >>> from morie.fn.aftwbl import aft_weibull
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(1500, 2))
    >>> mu = 1.0 + 0.7 * X[:, 0] - 0.4 * X[:, 1]
    >>> T = np.exp(mu + 0.6 * np.log(rng.exponential(1.0, 1500)))
    >>> C = rng.exponential(float(np.exp(mu).mean()) * 20, 1500)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = aft_residuals(aft_weibull(t, e, X))
    >>> cs = r["cox_snell"][e == 1]
    >>> bool(abs(float(cs.mean()) - 1.0) < 0.15)
    True

    Cox-Snell residuals are non-negative by construction.

    >>> bool(r["cox_snell"].min() >= 0.0)
    True

    Martingale residuals keep the Cox bound: at most 1, unbounded below.

    >>> bool(r["martingale"].max() <= 1.0 + 1e-9)
    True

    >>> aft_residuals({"beta": [1.0]})
    Traceback (most recent call last):
        ...
    ValueError: fit is missing 'time'; pass a result from one of the AFT fitters
    """
    for k in ("time", "event", "X", "beta", "log_sigma", "family"):
        if fit.get(k) is None:
            raise ValueError(
                f"fit is missing {k!r}; pass a result from one of the AFT fitters"
            )
    t = np.asarray(fit["time"], dtype=float)
    e = np.asarray(fit["event"], dtype=float)
    X = np.atleast_2d(np.asarray(fit["X"], dtype=float))
    beta = np.asarray(fit["beta"], dtype=float).ravel()
    sigma = float(np.exp(fit["log_sigma"]))
    fam = fit["family"]

    A = np.column_stack([np.ones(t.size), X]) if beta.size == X.shape[1] + 1 else X
    z = (np.log(np.maximum(t, 1e-300)) - A @ beta) / sigma
    _, log_surv = log_dens_surv(z, fam)
    cs = -log_surv                      # unit exponential under a correct model
    mart = e - cs
    inner = -2.0 * (mart + np.where(e > 0, e * np.log(np.maximum(e - mart, 1e-300)), 0.0))
    dev = np.sign(mart) * np.sqrt(np.maximum(inner, 0.0))
    return RichResult(
        title=f"AFT residuals ({fam})",
        summary_lines=[("n", int(t.size)), ("family", fam),
                       ("mean Cox-Snell (events)",
                        float(cs[e == 1].mean()) if np.any(e == 1) else float("nan"))],
        payload={
            "standardized": z, "cox_snell": cs, "martingale": mart,
            "deviance": dev, "event": e, "family": fam,
            "method": "aft_residuals",
        },
    )


def cheatsheet():
    return "aftres: Cox-Snell residuals are unit-exponential under ANY correct family -- one plot checks the lot"
