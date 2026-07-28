# morie.fn -- function file (rootcoder007/morie)
"""Lipschitz bound for LAD criteria."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_lad_lipschitz_bound"]


def kosorok_ch2_lad_lipschitz_bound(theta_1, theta_2, u, x=None):
    r"""Lipschitz-in-parameter bound for the least-absolute-deviation
    criterion:

    .. math:: |m_{\theta_1}(x) - m_{\theta_2}(x)|
              \le \|\theta_1 - \theta_2\| \cdot \|u\|.

    A Lipschitz-in-parameter class with a square-integrable envelope
    is Donsker, so this single bound is what puts the non-smooth LAD
    criterion inside the empirical-process machinery -- no
    differentiability of m required.

    Verifies the bound directly on the supplied covariates: returns
    the observed maximum ratio, which must not exceed 1.

    Parameters
    ----------
    theta_1, theta_2 : array-like
        The two parameter values.
    u : array-like, shape (n, p)
        Covariate rows.
    x : array-like, optional
        Responses; zeros if omitted (the bound is response-free).

    Returns
    -------
    RichResult
        keys: ``max_ratio``, ``bound_holds``, ``lhs_max``,
        ``rhs_max``, ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (Lipschitz classes; the LAD example).
    """
    t1 = np.atleast_1d(np.asarray(theta_1, dtype=float))
    t2 = np.atleast_1d(np.asarray(theta_2, dtype=float))
    if t1.shape != t2.shape:
        raise ValueError("theta_1 and theta_2 must have the same shape.")
    U = np.atleast_2d(np.asarray(u, dtype=float))
    if U.shape[1] != t1.size:
        U = U.T
    if U.shape[1] != t1.size:
        raise ValueError("u must have one column per parameter.")
    y = np.zeros(U.shape[0]) if x is None else np.asarray(x, dtype=float).ravel()
    if y.size != U.shape[0]:
        raise ValueError("x must have one entry per row of u.")
    lhs = np.abs(np.abs(y - U @ t1) - np.abs(y - U @ t2))
    rhs = float(np.linalg.norm(t1 - t2)) * np.linalg.norm(U, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rhs > 0, lhs / rhs, 0.0)
    return RichResult(
        payload={"max_ratio": float(np.max(ratio)),
                 "bound_holds": bool(np.max(ratio) <= 1.0 + 1e-9),
                 "lhs_max": float(lhs.max()), "rhs_max": float(rhs.max()),
                 "n": int(U.shape[0]),
                 "method": "|m_t1 - m_t2| <= ||t1 - t2|| ||u||, checked pointwise"}
    )


def cheatsheet():
    return "ksr056: Lipschitz-in-parameter + square envelope => Donsker, no smoothness"
