# morie.fn -- function file (rootcoder007/morie)
"""Continuity correction for integer-valued statistics."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_continuity_corr"]


def gibbons_continuity_corr(T, mu, sigma, cdf=None):
    r"""Section 1.2.13: when an integer-valued statistic is
    approximated by a normal, half a unit is moved toward the mean:

    .. math:: Z_{cc} = \frac{|T - \mu| - 0.5}{\sigma},

    matching the discrete mass at T to the continuous area between
    T - 1/2 and T + 1/2. Both the corrected and uncorrected z are
    returned; the correction always WEAKENS the evidence (smaller
    |z|), never strengthens it.

    Parameters
    ----------
    T : float
        Observed integer-valued statistic.
    mu, sigma : float
        Null mean and standard deviation.
    cdf : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``z_corrected``, ``z_uncorrected``, ``p_two_sided``,
        ``p_uncorrected``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 1.2.13.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    T, mu = float(T), float(mu)
    z0 = (T - mu) / sigma
    zc = max(abs(T - mu) - 0.5, 0.0) / sigma
    return RichResult(
        payload={
            "z_corrected": float(zc), "z_uncorrected": float(z0),
            "p_two_sided": float(2 * stats.norm.sf(zc)),
            "p_uncorrected": float(2 * stats.norm.sf(abs(z0))),
            "method": "Z_cc = (|T - mu| - 0.5)/sigma (Gibbons Ch. 1.2.13)",
        }
    )


def cheatsheet():
    return "gb_cc: half-unit toward the mean; always weakens, never strengthens"
