# morie.fn -- function file (rootcoder007/morie)
"""Variance of a sum of two uncorrelated random processes (Rangayyan Eq 3.14)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_variance_of_sum_uncorrelated"]


def rangayyan_ch3_variance_of_sum_uncorrelated(sigma_x, sigma_eta):
    r"""Variance of :math:`y = x + \eta` for uncorrelated :math:`x` and :math:`\eta`.

    .. math::

        E[(y - \mu_y)^2] = \sigma_y^2 = \sigma_x^2 + \sigma_\eta^2

    Parameters
    ----------
    sigma_x, sigma_eta : float or array-like
        Standard deviations of the signal and noise processes. These are
        **SDs, not signals**, and must be non-negative.

    Returns
    -------
    RichResult
        keys: ``variance``, ``sd``, ``sigma_x``, ``sigma_eta``, ``method``.

    Raises
    ------
    ValueError
        If either standard deviation is negative or non-finite.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.14), p. 96. Follows from Eq. (3.12) :math:`y(t)=x(t)+\eta(t)`
        and holds when :math:`x` and :math:`\eta` are uncorrelated; the book
        adds that then "their covariance and correlation coefficient are zero".

    Notes
    -----
    The hypothesis is not checkable from the arguments -- only SDs are passed,
    never the data -- so the caller owns it. For *correlated* processes the
    true variance of the sum exceeds this by :math:`2C_{x\eta}` (Eq. 3.21).
    """
    sx = np.asarray(sigma_x, dtype=float)
    se = np.asarray(sigma_eta, dtype=float)
    if not (np.all(np.isfinite(sx)) and np.all(np.isfinite(se))):
        raise ValueError(
            f"standard deviations must be finite; got sigma_x={sigma_x!r}, sigma_eta={sigma_eta!r}"
        )
    if np.any(sx < 0) or np.any(se < 0):
        raise ValueError(
            f"standard deviations must be non-negative; got sigma_x={sigma_x!r}, "
            f"sigma_eta={sigma_eta!r} -- these are SDs, not signal samples"
        )
    var = sx**2 + se**2
    scalar = var.ndim == 0
    return RichResult(
        payload={
            "variance": float(var) if scalar else var,
            "sd": float(np.sqrt(var)) if scalar else np.sqrt(var),
            "sigma_x": float(sx) if sx.ndim == 0 else sx,
            "sigma_eta": float(se) if se.ndim == 0 else se,
            "method": "variance of a sum of uncorrelated processes (Rangayyan Eq 3.14)",
        }
    )


def cheatsheet():
    return "rng014: sigma_y^2 = sigma_x^2 + sigma_eta^2 (Rangayyan Eq 3.14)."
