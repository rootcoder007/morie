# morie.fn -- function file (rootcoder007/morie)
"""Correlation coefficient as normalised covariance (Rangayyan Eq 3.22)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_correlation_coefficient"]


def rangayyan_ch3_correlation_coefficient(C_xy, sigma_x, sigma_y):
    r"""Normalise a covariance into a correlation coefficient.

    .. math::

        \rho_{xy} = \frac{C_{xy}}{\sigma_x \sigma_y}

    Parameters
    ----------
    C_xy : float
        Covariance :math:`C_{xy} = E[(x-\mu_x)(y-\mu_y)]` (Eq. 3.21).
    sigma_x, sigma_y : float
        Standard deviations. Both must be strictly positive -- with a
        degenerate process the coefficient is undefined, not zero.

    Returns
    -------
    RichResult
        keys: ``value`` (:math:`\rho_{xy}`), ``C_xy``, ``sigma_x``,
        ``sigma_y``, ``method``.

    Raises
    ------
    ValueError
        If either SD is non-positive, or if the inputs imply
        :math:`|\rho_{xy}| > 1`, which the Cauchy-Schwarz inequality forbids.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*. Wiley.
        Eq. (3.22), p. 98, "with :math:`-1 \le \rho_{xy} \le +1`". Covariance
        is Eq. (3.21) on the same page.

    Notes
    -----
    The book states the bound :math:`-1 \le \rho_{xy} \le +1` as part of the
    definition, so a result outside it means the *inputs* are inconsistent --
    some covariance and SDs that cannot come from the same pair of processes.
    That is raised rather than returned, because a "correlation" of 1.4 is not
    a number any caller can use.
    """
    c = float(C_xy)
    sx = float(sigma_x)
    sy = float(sigma_y)
    if not np.isfinite(c):
        raise ValueError(f"C_xy must be finite; got {C_xy!r}")
    if not (sx > 0 and sy > 0) or not (np.isfinite(sx) and np.isfinite(sy)):
        raise ValueError(
            f"standard deviations must be finite and strictly positive; "
            f"got sigma_x={sigma_x!r}, sigma_y={sigma_y!r}. rho is undefined "
            "for a degenerate process."
        )
    rho = c / (sx * sy)
    if abs(rho) > 1.0:
        raise ValueError(
            f"C_xy={c!r} with sigma_x={sx!r}, sigma_y={sy!r} gives rho={rho!r}, "
            "outside the [-1, +1] range Eq. (3.22) states. Cauchy-Schwarz "
            "forbids |C_xy| > sigma_x*sigma_y, so these inputs are inconsistent."
        )
    return RichResult(
        payload={
            "value": rho,
            "C_xy": c,
            "sigma_x": sx,
            "sigma_y": sy,
            "method": "correlation coefficient rho = C_xy/(sigma_x sigma_y) (Rangayyan Eq 3.22)",
        }
    )


def cheatsheet():
    return "rng022: rho_xy = C_xy/(sigma_x sigma_y) (Rangayyan Eq 3.22)."
