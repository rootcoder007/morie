"""F-function (empty space function)."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import as_points, as_region, intensity

__all__ = ["schabenberger_f_function"]


def schabenberger_f_function(points, region=None, r=None, n_grid=40):
    r"""
    F-function: the empty space (point-to-nearest-event) distribution.

    :math:`F(r)` is the CDF of the distance from an ARBITRARY location
    to the nearest event, in contrast to :math:`G`, which measures from
    an arbitrary EVENT. It is estimated here from a regular grid of
    sample locations over the region.

    Under CSR the two have the same form,

    .. math::

        F(r) = 1 - e^{-\lambda \pi r^2}

    but they respond to departures in opposite directions: clustering
    leaves large empty gaps, so :math:`\hat{F}` falls BELOW the CSR
    curve while :math:`\hat{G}` rises above it.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices; defaults to the
        bounding box of ``points``.
    r : array-like, optional
        Distances at which to evaluate the CDF.
    n_grid : int, default 40
        Sample grid is ``n_grid`` x ``n_grid``.

    Returns
    -------
    RichResult
        ``r``, ``f`` (empirical), ``f_csr`` (theoretical),
        ``empty_space_distances``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.3.4, pp. 97-98.
    """
    p = as_points(points)
    xmin, ymin, xmax, ymax = as_region(region, p)
    gx = np.linspace(xmin, xmax, int(n_grid))
    gy = np.linspace(ymin, ymax, int(n_grid))
    grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    d = np.linalg.norm(grid[:, None, :] - p[None, :, :], axis=-1).min(axis=1)
    if r is None:
        r = np.linspace(0.0, float(d.max()), 25)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    f = np.array([(d <= y).sum() / d.size for y in r], dtype=float)
    lam = intensity(p, (xmin, ymin, xmax, ymax))
    return RichResult(
        title="F-function (empty space)",
        summary_lines=[("grid points", int(d.size)),
                       ("mean empty-space distance", float(d.mean()))],
        payload={"r": r, "f": f, "f_csr": 1.0 - np.exp(-lam * np.pi * r**2),
                 "empty_space_distances": d, "lambda_est": lam},
    )


def cheatsheet():
    return "spffun: F(r), empty-space CDF; falls BELOW the CSR curve when clustered."
