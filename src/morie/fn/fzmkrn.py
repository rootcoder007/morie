# morie.fn -- function file (rootcoder007/morie)
"""Fourth-order (Muller) kernel for quantile estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_muller_kernel", "fauzi_muller_fourth_order_kernel"]


def fauzi_muller_kernel(u):
    r"""The fourth-order Muller kernel used for quantile estimation
    (Fauzi Sec. 3.4):

    .. math:: K_4(u) = \frac{3 - u^2}{2}\,\phi(u),

    built from the Gaussian so that
    :math:`\int u^2 K_4 = 0` while :math:`\int K_4 = 1`.

    The concrete instance of :mod:`morie.fn.fzkoc` that the chapter
    actually uses. Its second moment vanishes by construction, so the
    quantile estimator's bias is :math:`O(h^4)` rather than
    :math:`O(h^2)`, and it is negative for :math:`|u| > \sqrt3` --
    which is exactly the price the order condition forces.

    Parameters
    ----------
    u : array-like
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``u``, ``K``, ``mu0``, ``mu2``, ``mu4``,
        ``negative_beyond``, ``bias_order``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Sec. 3.4; Muller (1991).
    """
    from ._fauzi import muller_order_m

    uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    K = muller_order_m(uv, 4)
    grid = np.linspace(-10, 10, 4001)
    Kg = muller_order_m(grid, 4)
    return RichResult(payload={
        "u": uv, "K": K,
        "mu0": float(np.trapezoid(Kg, grid)),
        "mu2": float(np.trapezoid(grid ** 2 * Kg, grid)),
        "mu4": float(np.trapezoid(grid ** 4 * Kg, grid)),
        "negative_beyond": float(np.sqrt(3.0)),
        "bias_order": "O(h^4)",
        "method": "Fourth-order Muller kernel (3 - u^2)phi(u)/2; mu_2 = 0 by construction"})


def cheatsheet():
    return "fzmkrn: (3 - u^2)phi(u)/2 -- mu_2 = 0, and negative past |u| = sqrt(3)"


#: Catalogue alias for :func:`fauzi_muller_kernel`.
fauzi_muller_fourth_order_kernel = fauzi_muller_kernel
