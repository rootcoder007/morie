# morie.fn -- function file (rootcoder007/morie)
"""Bias and variance of the boundary-free KDE (Theorem 5.5)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkdebv", "fauzi_thm5_5_bdfree_kde_bv"]


def bfkdebv(n, h, density, c2, dg, mu2=1.0, rk=None):
    r"""Bias and variance of the boundary-free KDE (Theorem 5.5).

    Theorem 5.5, Eqs. (5.10)-(5.11):

    .. math::
        \mathrm{Bias}[\tilde f_X(x)] &=
            \frac{h^2c_2(x)}{2g'(g^{-1}(x))}\mu_2(K) + o(h^2), \\
        \mathrm{Var}[\tilde f_X(x)] &=
            \frac{f_X(x)}{nhg'(g^{-1}(x))}\int K^2(v)dv
            + o\!\Big(\frac1{nh}\Big).

    Both carry :math:`1/g'` -- and it is the SAME power in both, which is
    why the transformation does not change the bias-variance tradeoff's
    shape, only its constants. The MSE-optimal bandwidth is still
    :math:`O(n^{-1/5})` and the optimal MSE still :math:`O(n^{-4/5})`.

    The contrast with Theorem 5.2 is the point. There the variance was
    :math:`O(1/n) - O(h/n)` and smoothing HELPED; here it is
    :math:`O(1/(nh))` and smoothing hurts, so a bandwidth must be traded
    against bias in the usual way. A distribution function is estimable at
    the parametric rate; a density is not.

    :math:`\int K^2(v)dv` defaults to the Gaussian
    :math:`1/(2\sqrt\pi)`.

    Parameters
    ----------
    n : int
        Sample size.
    h : float
        Bandwidth.
    density : float
        ``f_X(x)``.
    c2 : float
        The Theorem 5.5 coefficient.
    dg : float
        ``g'(g^{-1}(x))``, strictly positive.
    mu2 : float, default 1.0
        ``int v^2 K(v) dv``.
    rk : float, optional
        ``int K^2(v) dv``; defaults to the Gaussian ``1/(2 sqrt(pi))``.

    Returns
    -------
    RichResult
        Keys ``bias``, ``variance``, ``se``, ``mse``, ``hopt``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.5, Eqs. (5.10)-(5.11).
    """
    n = int(n)
    h = float(h)
    dg = float(dg)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if dg <= 0:
        raise ValueError("g' must be positive; g is an increasing bijection (D4).")
    if rk is None:
        rk = 1.0 / (2.0 * np.sqrt(np.pi))
    rk = float(rk)
    bias = h * h * float(c2) / (2.0 * dg) * float(mu2)
    var = float(density) * rk / (n * h * dg)
    lead = float(c2) * float(mu2) / (2.0 * dg)
    if lead != 0.0 and float(density) > 0:
        hopt = float(
            (float(density) * rk / (dg * 4.0 * lead ** 2 * n)) ** 0.2
        )
    else:
        hopt = float("nan")
    return RichResult(
        payload={
            "bias": float(bias),
            "variance": float(var),
            "se": float(np.sqrt(var)),
            "mse": float(bias * bias + var),
            "hopt": hopt,
            "h": h,
            "n": n,
            "method": "boundary-free KDE bias and variance (Theorem 5.5)",
        }
    )


fauzi_thm5_5_bdfree_kde_bv = bfkdebv


def cheatsheet():
    return "fzt55: Thm 5.5: variance O(1/(nh)) -- unlike the df estimators, smoothing HURTS here"


# CANONICAL TEST
# >>> r = bfkdebv(n=100, h=0.3, density=0.4, c2=-0.1, dg=1.0)
# >>> r['variance'] > 0 and r['hopt'] > 0
# True
