# morie.fn -- function file (rootcoder007/morie)
"""Empirical distribution of regression residuals."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_residual_edf"]


def kosorok_residual_edf(y, z, beta, t=None):
    r"""Empirical distribution function of the residuals in a linear
    regression (Kosorok Eq. 1.2, p. 4):

    .. math:: \hat F(t) = n^{-1}\sum_{i=1}^{n}
              \mathbf 1\{Y_i - \hat\beta' Z_i \le t\}.

    The book opens with this because it is the simplest object where
    the estimated parameter INSIDE the indicator matters. If
    :math:`\beta` were known, :math:`\hat F` would be an ordinary
    empirical distribution function and Donsker's theorem would
    apply directly. It is not, and plugging in :math:`\hat\beta`
    adds a term to the limit: the process
    :math:`\sqrt n(\hat F - F)` is NOT the standard Brownian
    bridge, and treating it as one gives wrong standard errors.

    That correction is why empirical process theory is needed rather
    than the classical central limit theorem, and it is the reason
    ``limit_is_brownian_bridge`` is returned as False.

    Parameters
    ----------
    y : array-like, shape (n,)
        Responses.
    z : array-like, shape (n,) or (n, p)
        Covariates.
    beta : array-like, shape (p,)
        Estimated coefficients.
    t : array-like, optional
        Evaluation points; the sorted residuals otherwise.

    Returns
    -------
    RichResult
        keys: ``t``, ``F_hat``, ``residuals``,
        ``limit_is_brownian_bridge`` (False), ``correction_note``,
        ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 1, Eq. (1.2), p. 4.
    """
    yv = np.asarray(y, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != yv.size:
        Z = Z.T
    if Z.shape[0] != yv.size:
        raise ValueError("z must have one row per entry of y.")
    b = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    if b.size != Z.shape[1]:
        raise ValueError(f"beta has {b.size} entries for {Z.shape[1]} columns.")
    resid = yv - Z @ b
    tv = np.sort(resid) if t is None else \
        np.atleast_1d(np.asarray(t, dtype=float))
    F = np.array([float(np.mean(resid <= v)) for v in tv])
    return RichResult(payload={
        "t": tv, "F_hat": F, "residuals": resid,
        "limit_is_brownian_bridge": False,
        "correction_note": "plugging in beta-hat adds a term to the limit of "
                           "sqrt(n)(F_hat - F); it is not the standard bridge",
        "n": int(yv.size),
        "method": "Residual empirical df (Eq. 1.2); the estimated beta inside the indicator changes the limit"})


def cheatsheet():
    return "ksr021: beta-hat inside the indicator means the limit is NOT a Brownian bridge"
