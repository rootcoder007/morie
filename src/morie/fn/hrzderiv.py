# morie.fn -- function file (rootcoder007/morie)
"""Kernel density derivative."""

import numpy as np

from ._horowitz import kde_deriv
from ._richresult import RichResult

__all__ = ["hrz_density_derivative"]


def hrz_density_derivative(x, grid=None, h=None, kernel_name="gaussian", r=1):
    r"""Kernel estimate of the density derivative (Horowitz Ch. 2):

    .. math:: \hat f'(x) = -\frac{1}{nh^2}\sum_i
              K'\!\left(\frac{x - X_i}{h}\right),

    converging at :math:`O_p(n^{-r/(2r+3)})` for the rth derivative --
    strictly slower than the density itself. The bandwidth must be
    WIDER than the density-optimal one; reusing that bandwidth
    undersmooths badly, which is the standard mistake here.

    Parameters
    ----------
    x : array-like
        Sample.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth; a derivative-appropriate rule if omitted.
    kernel_name : str
        Kernel.
    r : int, default 1
        Derivative order, used for the reported rate.

    Returns
    -------
    RichResult
        keys: ``grid``, ``derivative``, ``bandwidth``,
        ``rate_exponent``, ``r``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (derivative estimation).
    """
    r = int(r)
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}.")
    g, d, hh = kde_deriv(x, grid=grid, h=h, name=kernel_name)
    return RichResult(payload={"grid": g, "derivative": d, "bandwidth": hh,
                               "rate_exponent": -r / (2.0 * r + 3.0), "r": r,
                               "method": "f-hat'(x) via K'; needs a WIDER bandwidth than the density"})


def cheatsheet():
    return "hrzderiv: reusing the density bandwidth undersmooths the derivative"
