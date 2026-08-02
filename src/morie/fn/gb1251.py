# morie.fn -- function file (rootcoder007/morie)
"""Kendall partial tau controlling for a confounder z."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_partial_tau"]


def gibbons_partial_tau(x, y, z):
    r"""Kendall's partial rank correlation.

    .. math:: \tau_{xy \cdot z} = \frac{\tau_{xy} - \tau_{xz}\tau_{yz}}
              {\sqrt{(1 - \tau_{xz}^2)(1 - \tau_{yz}^2)}},

    the rank analogue of the partial product-moment correlation: it
    removes the part of the x-y association attributable to their
    common association with z, without assuming linearity or
    normality. A partial tau far below the marginal tau is the
    rank-based signature of confounding by z.

    Caveat: unlike the Gaussian partial correlation, Kendall's partial
    tau is *not* a conditional-independence measure -- it does not go
    to zero under exact conditional independence (in a linear-Gaussian
    fork it settles near 0.2 while the marginal tau is 0.6). Read it
    as a shrinkage diagnostic, not as a test of X independent of Y
    given Z.

    Parameters
    ----------
    x, y, z : array-like, shape (n,)
        Observations; ties are handled by tau-b.

    Returns
    -------
    RichResult
        keys: ``partial_tau``, ``tau_xy``, ``tau_xz``, ``tau_yz``,
        ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2011). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Sec. 12.5 (Kendall's
    partial tau).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = x.size
    if not (y.size == n and z.size == n):
        raise ValueError("x, y, z must have equal length.")
    if n < 4:
        raise ValueError(f"need at least 4 observations, got {n}.")

    t_xy = float(stats.kendalltau(x, y).statistic)
    t_xz = float(stats.kendalltau(x, z).statistic)
    t_yz = float(stats.kendalltau(y, z).statistic)
    den = np.sqrt((1 - t_xz**2) * (1 - t_yz**2))
    if den <= 0:
        raise ValueError("a marginal tau with z is +/-1; the partial tau is undefined.")

    return RichResult(
        payload={
            "partial_tau": float((t_xy - t_xz * t_yz) / den),
            "tau_xy": t_xy,
            "tau_xz": t_xz,
            "tau_yz": t_yz,
            "n": int(n),
            "method": "Kendall partial tau controlling for z",
        }
    )


def cheatsheet():
    return "gb1251: tau_xy.z = (t_xy - t_xz t_yz) / sqrt((1-t_xz^2)(1-t_yz^2))"
