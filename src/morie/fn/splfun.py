"""L-function: variance-stabilized K-function."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import as_region

__all__ = ["schabenberger_l_function"]


def schabenberger_l_function(points, lambda_est=None, r=None, region=None,
                             correction="border"):
    r"""
    L-function, the variance-stabilised K-function.

    .. math::

        \hat{L}(h) = \sqrt{\hat{K}(h)/\pi}

    Under CSR :math:`K(h) = \pi h^2`, so :math:`L(h) = h` exactly and
    :math:`L(h) - h` is a horizontal reference line at zero. The book
    recommends plotting :math:`\hat{L}(h) - h` against :math:`h`:
    clustering appears as positive values at short distances, regularity
    as negative ones. K rises quickly with h and can look erratic at
    large distances, which is what the square root stabilises.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    lambda_est : float, optional
        Intensity; estimated from ``region`` when omitted.
    r : array-like, optional
        Distances at which to evaluate.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices.
    correction : {'border', 'none'}
        Edge correction passed through to the K-function.

    Returns
    -------
    RichResult
        ``r``, ``l``, ``l_minus_r`` (the CSR-centred curve), ``k``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 3.4.2, p. 103.
    """
    from .spkfun import schabenberger_k_function

    reg = as_region(region, points)
    kr = schabenberger_k_function(points, lambda_est, r, reg, correction)
    rr, k = kr["r"], kr["k"]
    with np.errstate(invalid="ignore"):
        ell = np.sqrt(np.maximum(k, 0.0) / np.pi)
    return RichResult(
        title="L-function (variance-stabilised K)",
        summary_lines=[("lambda", kr["lambda_est"]), ("correction", correction)],
        payload={"r": rr, "l": ell, "l_minus_r": ell - rr, "k": k,
                 "lambda_est": kr["lambda_est"]},
    )


def cheatsheet():
    return "splfun: L(h)=sqrt(K(h)/pi); L(h)-h is 0 under CSR."
