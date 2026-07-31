"""Semivariogram definition as half the mean squared difference."""

import numpy as np

from ._richresult import RichResult
from ._schab_vario import empirical_semivariogram

__all__ = ["schabenberger_semivariogram_def"]


def schabenberger_semivariogram_def(coords, z, n_bins=15, max_dist=None):
    r"""
    Empirical semivariogram: half the mean squared difference.

    .. math::

        \gamma(h) = \tfrac{1}{2}\,\mathrm{E}\!\left[(Z(s) - Z(s+h))^2\right]

    estimated by Matheron's method of moments over lag bins,

    .. math::

        \hat\gamma(h) = \frac{1}{2|N(h)|}\sum_{N(h)} (Z(s_i) - Z(s_j))^2

    The factor of one half is what makes :math:`\gamma` comparable to a
    variance rather than to a squared difference.

    Parameters
    ----------
    coords : array-like
        Coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    n_bins : int, default 15
        Number of lag bins.
    max_dist : float, optional
        Largest lag retained. Defaults to half the maximum pair distance,
        the usual rule for keeping bin counts usable.

    Returns
    -------
    RichResult
        ``lag``, ``gamma`` and ``n_pairs`` per bin.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 1.4.3 / Ch. 4.
    """
    lag, gam, cnt = empirical_semivariogram(coords, z, n_bins, max_dist)
    return RichResult(
        title="Empirical semivariogram (method of moments)",
        summary_lines=[("bins", int(n_bins)), ("pairs used", int(cnt.sum()))],
        payload={"lag": lag, "gamma": gam, "n_pairs": cnt},
    )


def cheatsheet():
    return "spsemv: empirical semivariogram, half the mean squared difference."
