# morie.fn -- function file (rootcoder007/morie)
"""Boundary-free kernel density estimator via bijective transformation g."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_boundary_free_kde"]


def fauzi_boundary_free_kde(x, grid=None, h=None, transform="log"):
    r"""Boundary-free kernel density estimator (Fauzi Ch. 4):

    .. math:: \tilde f_X(t) = \frac{1}{n\,h\,g'(g^{-1}(t))}
              \sum_{i=1}^{n}
              K\!\left(\frac{g^{-1}(t) - g^{-1}(X_i)}{h}\right).

    A bijection :math:`g` carries the bounded support onto the whole
    line, an ordinary symmetric kernel is applied THERE, and the
    result is mapped back. Because the transformed problem has no
    boundary, the O(h) edge bias of
    :mod:`morie.fn.fzkde` never arises.

    The factor :math:`1/g'(g^{-1}(t))` is the change-of-variables
    Jacobian and it is not optional: without it the result is a
    density on the transformed scale, not on the original one, and
    integrates to something other than 1. It is also exactly the
    piece the library's distilled text file drops, which is why this
    module was written from the PDF.

    Parameters
    ----------
    x : array-like
        Sample inside the support of ``g``.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth on the TRANSFORMED scale.
    transform : {"log", "identity"}
        The bijection; ``log`` maps (0, inf) to the line.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``transform``,
        ``jacobian`` (= 1/g'(g^{-1}(t))), ``g_prime``, ``mass``, ``boundary_bias_order`` ("O(h^2)
        everywhere, including the boundary"), ``n``, ``method``.
    References
    ----------
    Fauzi, R. R. and Maesono, Y. *Statistical Inference Based on
    Kernel Distribution Function Estimators*. Springer, 2023.
    Ch. 4, the boundary-free kernel density estimator. Transcribed from the PDF: the distilled text file in the
    reference library omits the Jacobian factor and truncates (4.24).
    """
    from ._fauzi import boundary_free_transform, kernel_K

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    tr = boundary_free_transform(transform)
    lo, hi = tr["support"]
    if np.any(xv <= lo) or np.any(xv >= hi):
        raise ValueError(f"the sample must lie strictly inside {tr['support']} "
                         f"for the {tr['name']} transformation.")
    z = tr["g_inv"](xv)
    hh = float(np.std(z, ddof=1) * n ** -0.2) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(float(np.quantile(xv, 0.02)),
                    float(np.quantile(xv, 0.98)), 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    if np.any(g <= lo) or np.any(g >= hi):
        raise ValueError("the grid must lie strictly inside the support.")
    gz = tr["g_inv"](g)
    # the change-of-variables factor itself, 1/g'(g^{-1}(t)) -- the
    # quantity the density is multiplied by, so that is what the
    # "jacobian" key reports
    g_prime = tr["dg"](gz)
    jac = 1.0 / g_prime
    dens = kernel_K((gz[:, None] - z[None, :]) / hh).sum(axis=1) * jac / (n * hh)
    return RichResult(payload={
        "grid": g, "density": dens, "bandwidth": hh,
        "transform": tr["name"], "jacobian": jac, "g_prime": g_prime,
        "mass": float(np.trapezoid(dens, g)),
        "boundary_bias_order": "O(h^2) everywhere, including the boundary",
        "jacobian_note": "1/g'(g^{-1}(t)) is the change-of-variables factor; "
                         "without it the result is a density on the "
                         "transformed scale, not the original one",
        "n": int(n),
        "method": "Boundary-free KDE by bijection (Ch. 4); no boundary exists on the transformed scale"})


def cheatsheet():
    return "fzbfkd: the 1/g'(g^{-1}(t)) Jacobian is mandatory -- without it the density is on the wrong scale"
