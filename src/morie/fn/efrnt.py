# morie.fn -- function file (rootcoder007/morie)
"""Efron tie correction for the Cox partial likelihood."""

from __future__ import annotations

from ._surv import cox_fit, prepare
from .breslot import _cox_result

__all__ = ["efron_tie_correction"]


def efron_tie_correction(time, event, X, **kwargs):
    r"""Fit a Cox model using Efron's handling of tied event times.

    Where Breslow uses the full risk set for every tied event, Efron averages
    the tied contributions out of the denominator as the events are taken in
    turn:

    .. math::
        L_t = \prod_{l=0}^{d-1}
            \frac{e^{\beta^\top x_{(l)}}}
                 {\sum_{k \in R_t} e^{\beta^\top x_k}
                  - \frac{l}{d}\sum_{i \in D_t} e^{\beta^\top x_i}} .

    The :math:`l/d` term is the correction: by the time the :math:`l`-th of
    :math:`d` tied events is counted, a fraction :math:`l/d` of the tied
    subjects has on average already left the risk set. This approximates the
    exact discrete-time likelihood closely at a fraction of its cost -- the
    exact version sums over all :math:`\binom{n_R}{d}` orderings and is
    intractable for more than a handful of ties.

    With no ties Efron and Breslow are algebraically identical, which the
    doctest checks.

    Parameters
    ----------
    time : array-like
        Observed follow-up times.
    event : array-like
        1 for the event, 0 for right-censoring.
    X : array-like
        Covariates ``(n, p)``.
    **kwargs
        Passed to the Newton-Raphson fitter.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``z``, ``p_value``, ``hazard_ratio``, ``loglik``,
        ``n_ties``, ``converged``.

    References
    ----------
    Efron, B. (1977). The efficiency of Cox's likelihood function for censored
        data. *JASA*, 72(359), 557-565.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(2.0, 400)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> b = efron_tie_correction(t, e, X)["beta"]
    >>> [bool(abs(b[i] - v) < 0.25) for i, v in enumerate([0.8, -0.5])]
    [True, True]

    Without ties the two corrections coincide exactly -- there is nothing to
    correct.

    >>> from morie.fn.breslot import breslow_tie_correction
    >>> a = efron_tie_correction(t, e, X)["beta"]
    >>> c = breslow_tie_correction(t, e, X)["beta"]
    >>> bool(np.allclose(a, c, atol=1e-8))
    True

    The hazard ratio is the exponentiated coefficient, which is how a Cox fit
    is reported.

    >>> r = efron_tie_correction(t, e, X)
    >>> bool(np.allclose(r["hazard_ratio"], np.exp(r["beta"])))
    True
    """
    t, e, Xm = prepare(time, event, X)
    beta, ll, I, U, it, conv = cox_fit(t, e, Xm, ties="efron", **kwargs)
    return _cox_result(t, e, beta, ll, I, it, conv, "Efron",
                       "efron_tie_correction", X=Xm)


def cheatsheet():
    return "efrnt: l/d correction approximates the exact discrete likelihood cheaply; == Breslow when no ties"
