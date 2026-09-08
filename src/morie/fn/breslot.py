# morie.fn -- function file (rootcoder007/morie)
"""Breslow tie correction for the Cox partial likelihood."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["breslow_tie_correction"]


def breslow_tie_correction(time, event, X, **kwargs):
    r"""Fit a Cox model using Breslow's handling of tied event times.

    Breslow treats each of :math:`d` events tied at time :math:`t` as though it
    occurred alone against the *full* risk set:

    .. math::
        L_t = \frac{\exp(\beta^\top \sum_{i \in D_t} x_i)}
                   {\left(\sum_{k \in R_t} e^{\beta^\top x_k}\right)^{d}} .

    That over-counts the denominator, because in truth each successive event
    faces a risk set already depleted by the earlier ones. The bias pulls
    coefficients **toward zero** and grows with the number of ties, so on
    coarsely recorded data -- survival in whole months, say -- it can be
    substantial.

    It is kept because it is fast and it is the default in several widely used
    packages, so reproducing a published Cox fit sometimes requires it. For new
    analyses prefer :func:`~morie.fn.efrnt.efron_tie_correction`.

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
    Breslow, N. (1974). Covariance analysis of censored survival data.
        *Biometrics*, 30(1), 89-99.
    Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*,
        34(2), 187-202.

    Examples
    --------
    Coefficients are recovered on data simulated from the model.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(2.0, 400)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> b = breslow_tie_correction(t, e, X)["beta"]
    >>> [bool(abs(b[i] - v) < 0.25) for i, v in enumerate([0.8, -0.5])]
    [True, True]

    With heavy ties the Breslow estimate is attenuated relative to Efron --
    the documented direction of the bias.

    >>> from morie.fn.efrnt import efron_tie_correction
    >>> tc = np.ceil(t * 3) / 3                      # coarse grid -> many ties
    >>> bb = breslow_tie_correction(tc, e, X)["beta"][0]
    >>> be = efron_tie_correction(tc, e, X)["beta"][0]
    >>> bool(bb < be)
    True

    >>> breslow_tie_correction([1.0, 2.0], [0.0, 2.0], [[1.0], [2.0]])
    Traceback (most recent call last):
        ...
    ValueError: event must be 0 (censored) or 1 (event)
    """
    t, e, Xm = prepare(time, event, X)
    beta, ll, I, U, it, conv = cox_fit(t, e, Xm, ties="breslow", **kwargs)
    return _cox_result(t, e, beta, ll, I, it, conv, "Breslow",
                       "breslow_tie_correction", X=Xm)


def _cox_result(t, e, beta, ll, I, it, conv, label, method, X=None):
    from ._stats_core import norm

    try:
        cov = np.linalg.inv(I)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov, se = None, np.full(beta.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    ev = t[e == 1]
    n_ties = int(ev.size - np.unique(ev).size)
    return RichResult(
        title=f"Cox model ({label} ties)",
        summary_lines=[("n", int(t.size)), ("events", int(e.sum())),
                       ("tied events", n_ties), ("loglik", ll)],
        warnings=[] if conv else ["Newton-Raphson did not converge"],
        payload={
            "beta": beta, "se": se, "z": z,
            "p_value": 2 * norm.sf(np.abs(z)),
            "hazard_ratio": np.exp(beta), "loglik": ll, "cov": cov,
            "information": I, "n_ties": n_ties, "n_events": int(e.sum()),
            "n": int(t.size), "n_iter": it, "converged": conv,
            "ties": label.lower(), "method": method,
            # Carried so the residual modules can work from a fit alone.
            "time": t, "event": e, "X": X,
        },
    )


def cheatsheet():
    return "breslot: Breslow ties over-count the risk set, attenuating beta toward 0; prefer Efron"
