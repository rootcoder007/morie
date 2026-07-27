# morie.fn -- function file (rootcoder007/morie)
"""Unit nonresponse propensity weighting."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit
from .nonresp import nonresponse_adjustment

__all__ = ["unit_nonresponse"]


def unit_nonresponse(respondents, frame, X, y=None):
    r"""Estimate response propensities over the frame and reweight.

    Fits a logistic regression of the response indicator on the frame
    covariates, then gives each respondent the weight
    :math:`w_i = d_i / \hat\varphi(x_i)` where :math:`d_i` is the
    design weight (taken as 1 if the frame carries none).

    Parameters
    ----------
    respondents : array-like of {0, 1}, shape (n,)
        Response indicator over the frame rows.
    frame : array-like, shape (n,) or None
        Design weights for the frame; None means equal weights.
    X : array-like, shape (n, p) or (n,)
        Frame covariates used to model response.
    y : array-like, shape (n,), optional
        Outcome recorded for respondents (values at nonrespondent rows
        are ignored). If given, the propensity-weighted Hajek mean is
        reported.

    Returns
    -------
    RichResult
        keys: ``propensity`` (n,), ``weights`` (respondent rows get
        d/phi, nonrespondents 0), ``estimate`` (or None), ``se``,
        ``ess``, ``n``, ``n_respondents``, ``method``.

    References
    ----------
    Little, R. J. & Vartivarian, S. (2005). Does weighting for
    nonresponse increase the variance of survey means? *Survey
    Methodology*, 31(2), 161-168.
    """
    r = np.asarray(respondents, dtype=float).ravel()
    if not np.all(np.isin(r, (0.0, 1.0))):
        raise ValueError("respondents must be a binary 0/1 indicator over the frame.")
    n = r.size
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but respondents has {n}.")
    d = np.ones(n) if frame is None else np.asarray(frame, dtype=float).ravel()
    if d.size != n or np.any(d <= 0):
        raise ValueError("frame design weights must be positive and match the frame length.")
    if r.sum() == 0 or r.sum() == n:
        raise ValueError("need both respondents and nonrespondents to fit a response model.")

    phi = np.clip(_logit_fit(X, r), 1e-6, 1.0)
    resp = r == 1
    w = np.zeros(n)
    w[resp] = d[resp] / phi[resp]

    est = se = ess = None
    if y is not None:
        y = np.asarray(y, dtype=float).ravel()
        sub = nonresponse_adjustment(y[resp], d[resp], phi[resp])
        est, se, ess = sub["estimate"], sub["se"], sub["ess"]

    return RichResult(
        payload={
            "propensity": phi,
            "weights": w,
            "estimate": est,
            "se": se,
            "ess": ess,
            "n": int(n),
            "n_respondents": int(resp.sum()),
            "method": "Unit nonresponse propensity weighting",
        }
    )


def cheatsheet():
    return "unitnr: logistic response propensity over frame, weights d/phi"
