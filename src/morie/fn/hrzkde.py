# morie.fn -- function file (rootcoder007/morie)
"""Univariate kernel density estimate."""

import numpy as np

from ._horowitz import kde, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_kde", "horowitz_appendix_kde"]


def hrz_kde(x, grid=None, h=None, kernel_name="gaussian"):
    r"""Kernel density estimate (Horowitz Ch. 2):

    .. math:: \hat f(x) = \frac{1}{nh}\sum_i
              K\!\left(\frac{x - X_i}{h}\right),

    with the MISE-optimal bandwidth of order :math:`n^{-1/5}`. The
    resulting rate :math:`n^{-2/5}` is slower than the parametric
    :math:`n^{-1/2}` and cannot be improved for a twice-differentiable
    density -- that gap is the reason the book builds root-n
    FUNCTIONALS of this object rather than using it directly.

    Parameters
    ----------
    x : array-like
        Sample.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth; Silverman's rule if omitted.
    kernel_name : {"gaussian", "epanechnikov", "uniform"}

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``rate_exponent``
        (-2/5), ``integrates_to``, ``n``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (kernel density estimation).
    """
    g, d, hh = kde(x, grid=grid, h=h, name=kernel_name)
    return RichResult(payload={"grid": g, "density": d, "bandwidth": hh,
                               "rate_exponent": -0.4,
                               "integrates_to": float(np.trapezoid(d, g)),
                               "n": int(np.asarray(x).size),
                               "method": "KDE with n^{-1/5} bandwidth; rate n^{-2/5}"})


def cheatsheet():
    return "hrzkde: n^{-2/5} rate is a ceiling, not a defect -- hence root-n functionals"


#: Catalogue alias for :func:`hrz_kde`.
horowitz_appendix_kde = hrz_kde
