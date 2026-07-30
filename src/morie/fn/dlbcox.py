# morie.fn -- function file (rootcoder007/morie)
"""DFBETA for a Cox model, fitted from raw data."""

from __future__ import annotations

from ._richresult import RichResult
from .coxdfb import cox_dfbeta_influence
from .efrnt import efron_tie_correction

__all__ = ["dfbeta_cox"]


def dfbeta_cox(time, event, X, ties="efron"):
    r"""Fit a Cox model and return its DFBETA influence measures in one call.

    Convenience front-end over :func:`~morie.fn.efrnt.efron_tie_correction`
    followed by :func:`~morie.fn.coxdfb.cox_dfbeta_influence`, for callers who
    have data rather than a fit. The influence arithmetic is identical.

    Parameters
    ----------
    time, event, X : array-like
        Survival data.
    ties : {"efron", "breslow"}
        Tie handling.

    Returns
    -------
    RichResult
        ``dfbeta``, ``dfbetas``, ``beta``, ``se``, ``most_influential``.

    References
    ----------
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(2.0, 200)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = dfbeta_cox(t, e, X)
    >>> r["dfbeta"].shape
    (200, 2)

    Agrees exactly with going through the fit object.

    >>> from morie.fn.efrnt import efron_tie_correction
    >>> from morie.fn.coxdfb import cox_dfbeta_influence
    >>> direct = cox_dfbeta_influence(efron_tie_correction(t, e, X))["dfbeta"]
    >>> bool(np.allclose(r["dfbeta"], direct))
    True
    """
    if ties == "efron":
        fit = efron_tie_correction(time, event, X)
    elif ties == "breslow":
        from .breslot import breslow_tie_correction

        fit = breslow_tie_correction(time, event, X)
    else:
        raise ValueError('ties must be "efron" or "breslow"')
    inf = cox_dfbeta_influence(fit)
    return RichResult(
        title="Cox DFBETA (fitted)",
        summary_lines=[("n", int(inf["dfbeta"].shape[0])),
                       ("most influential", int(inf["most_influential"]))],
        payload={
            "dfbeta": inf["dfbeta"], "dfbetas": inf["dfbetas"],
            "score_residuals": inf["score_residuals"],
            "beta": fit["beta"], "se": fit["se"],
            "max_influence": inf["max_influence"],
            "most_influential": inf["most_influential"],
            "method": "dfbeta_cox",
        },
    )


def cheatsheet():
    return "dlbcox: fit + dfbeta in one call; identical arithmetic to coxdfb on an existing fit"
