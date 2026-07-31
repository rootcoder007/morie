"""G-function: nearest-neighbour distance CDF."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import as_points, as_region, intensity, nn_distances

__all__ = ["schabenberger_g_function"]


def schabenberger_g_function(points, r=None, region=None):
    r"""
    G-function: the nearest-neighbour distance distribution.

    .. math::

        \hat{G}(y_0) = \frac{\#(y_i \le y_0)}{n}

    the empirical estimate of the probability that an event's
    nearest-neighbour distance is at most :math:`y_0`. Under CSR with
    intensity :math:`\lambda` the theoretical form is

    .. math::

        G(y) = 1 - e^{-\lambda \pi y^2}

    In a clustered pattern :math:`\hat{G}` sits ABOVE this curve at
    small :math:`y` (neighbours are unusually close); in a regular
    pattern it sits below.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    r : array-like, optional
        Distances at which to evaluate the CDF.
    region : array-like, optional
        Used only for the CSR reference intensity; defaults to the
        bounding box of ``points``.

    Returns
    -------
    RichResult
        ``r``, ``g`` (empirical), ``g_csr`` (theoretical),
        ``nn_distances``, ``mean_nn``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.3.4, pp. 97-98.
    """
    p = as_points(points)
    nn = nn_distances(p)
    if nn.size == 0:
        raise ValueError("at least two events are needed for a "
                         "nearest-neighbour distance")
    reg = as_region(region, p)
    if r is None:
        r = np.linspace(0.0, float(nn.max()), 25)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    g = np.array([(nn <= y).sum() / nn.size for y in r], dtype=float)
    lam = intensity(p, reg)
    return RichResult(
        title="G-function (nearest-neighbour distance CDF)",
        summary_lines=[("n events", int(p.shape[0])),
                       ("mean nn distance", float(nn.mean()))],
        payload={"r": r, "g": g, "g_csr": 1.0 - np.exp(-lam * np.pi * r**2),
                 "nn_distances": nn, "mean_nn": float(nn.mean()),
                 "lambda_est": lam},
    )


def cheatsheet():
    return "spgfun: G(y)=#(y_i<=y)/n; CSR reference 1-exp(-lambda pi y^2)."
