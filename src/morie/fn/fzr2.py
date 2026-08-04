# morie.fn -- function file (rootcoder007/morie)
"""The kernel constant r_2 of the bias-reduced KDFE variance."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfr2", "fauzi_r2_integral"]


def kdfr2(a, kernel="gaussian", lo=-8.0, hi=8.0, ngrid=4001):
    r"""The kernel constant r_2 of the bias-reduced KDFE variance.

    Eq. (2.10):

    .. math:: r_2 = \int_{-\infty}^{\infty} yK(y)\Big[W\!\big(\tfrac ya\big)
              + \tfrac1a W(y)K\!\big(\tfrac ya\big)\Big]dy.

    Wait -- read the bracket carefully. The book writes
    :math:`y[K(y)W(y/a) + a^{-1}W(y)K(y/a)]`, i.e. the two terms differ in
    WHICH factor is rescaled by ``a``. It is the cross term you get when
    you expand the variance of the linear combination
    :math:`\frac{a^2}{a^2-1}\hat F_h - \frac1{a^2-1}\hat F_{ah}` in
    Theorem 2.3, so it is not symmetric in the two bandwidths and must not
    be "simplified" into :math:`2r_1`.

    At :math:`a = 1` the two terms coincide and :math:`r_2 = 2r_1`, but
    :math:`a = 1` is excluded by Theorem 2.1 anyway (:math:`a^2-1` is a
    denominator).

    Integrated on a fixed trapezoid grid for any kernel including the
    Gaussian, because no closed form is stated in the book and inventing
    one would be worse than integrating.

    Parameters
    ----------
    a : float
        The second smoothing parameter of (2.5); ``a > 0``, ``a != 1``.
    kernel : {"gaussian"} or callable, default "gaussian"
    lo, hi : float, default -8.0, 8.0
        Quadrature limits.
    ngrid : int, default 4001
        Number of nodes; fixed, never adapted.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``a``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (2.10).
    """
    from . import _stats_core as stats

    a = float(a)
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}.")
    if a == 1.0:
        raise ValueError("a = 1 is excluded: (2.5) divides by a^2 - 1.")
    y = np.linspace(float(lo), float(hi), int(ngrid))
    if kernel == "gaussian":
        kfun = lambda t: stats.norm.pdf(t)
        wfun = lambda t: stats.norm.cdf(t)
    elif callable(kernel):
        kfun = lambda t: np.asarray([float(kernel(float(u))) for u in np.atleast_1d(t)], dtype=float)
        base = kfun(y)
        cum = np.concatenate(
            ([0.0], np.cumsum(np.diff(y) * (base[:-1] + base[1:]) / 2.0))
        )
        wfun = lambda t: np.interp(t, y, cum)
    else:
        raise ValueError('kernel must be "gaussian" or a callable K(y).')
    term = kfun(y) * wfun(y / a) + (1.0 / a) * wfun(y) * kfun(y / a)
    val = float(np.trapezoid(y * term, y))
    return RichResult(
        payload={
            "estimate": val,
            "a": a,
            "method": "r_2 cross-kernel constant (Eq. 2.10)",
        }
    )


fauzi_r2_integral = kdfr2


def cheatsheet():
    return "fzr2: r_2 -- the asymmetric cross-bandwidth kernel constant of Theorem 2.3 (Eq. 2.10)"


# CANONICAL TEST
# >>> r1 = 1 / (2 * (3.141592653589793 ** 0.5))
# >>> abs(kdfr2(a=1.0000001)['estimate'] - 2 * r1) < 1e-5  # a->1 limit
# True
