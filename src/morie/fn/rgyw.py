# morie.fn -- function file (rootcoder007/morie)
"""Yule-Walker AR estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_yule_walker"]


from .rgacf import rangayyan_acf_estimate


def rangayyan_yule_walker(x, order=4):
    r"""Yule-Walker autoregressive parameter estimation (Rangayyan
    Ch. 3):

    .. math:: \mathbf{R}_{xx}\,\mathbf{a} = -\mathbf{r},
              \qquad R_{xx}(i,j) = R_{xx}(|i-j|),

    a Toeplitz system solved for the AR coefficients. The BIASED
    autocorrelation is used deliberately: it is positive
    semi-definite, which guarantees the Toeplitz matrix is invertible
    and the fitted AR model is stable. The unbiased estimate can yield
    an unstable model, so it is the wrong input here even though it is
    the better estimate of the ACF itself.

    Parameters
    ----------
    x : array-like
        Signal.
    order : int, default 4
        AR order.

    Returns
    -------
    RichResult
        keys: ``a`` (AR coefficients), ``sigma2`` (innovation
        variance), ``order``, ``stable`` (all roots inside the unit
        circle), ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (Yule-Walker / AR modelling).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = int(order)
    if p < 1:
        raise ValueError(f"order must be at least 1, got {p}.")
    if x.size < p + 1:
        raise ValueError(f"need more than order = {p} samples, got {x.size}.")
    R = rangayyan_acf_estimate(x, max_lag=p, biased=True)["acf_biased"]
    Rm = np.array([[R[abs(i - j)] for j in range(p)] for i in range(p)])
    r = R[1 : p + 1]
    try:
        a = np.linalg.solve(Rm, -r)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(Rm, -r, rcond=None)[0]
    sigma2 = float(R[0] + a @ r)
    roots = np.roots(np.r_[1.0, a])
    return RichResult(payload={"a": a, "sigma2": sigma2, "order": p,
                               "stable": bool(np.all(np.abs(roots) < 1.0)),
                               "reflection_roots": roots,
                               "method": "Toeplitz Yule-Walker on the BIASED ACF (guarantees stability)"})


def cheatsheet():
    return "rgyw: biased ACF here is a feature -- it keeps the AR model stable"
