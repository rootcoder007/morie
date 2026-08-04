# morie.fn -- function file (rootcoder007/morie)
"""Variance of the bias-reduced KDFE (Theorem 2.3)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gekdfvar", "fauzi_thm2_3_var_brdkdfe"]


def gekdfvar(n, h, a, fx, density, r1=None, r2=None):
    r"""Variance of the bias-reduced KDFE (Theorem 2.3).

    Theorem 2.3:

    .. math:: \mathrm{Var}[\tilde F_X(x)] = \frac{F_X(1-F_X)}n
              - \frac hn\Big[\frac{2(a^4+1)}{(a^2-1)^2}r_1 + r_2\Big]
              f_X(x) + o\!\big(\tfrac hn\big),

    with :math:`r_1` from (2.9) and :math:`r_2` from (2.10).

    The order does not change from the plain KDFE -- it cannot, since
    :math:`\tilde F_X` linearises to a fixed linear combination of two
    KDFEs -- but the CONSTANT does, and the book's claim is that it is
    smaller. That is the whole content: bias improved from
    :math:`O(h^2)` to :math:`O(h^4)` at no cost in variance ORDER, and a
    gain in its constant.

    The bracket blows up as :math:`a\to1`: :math:`(a^2-1)^2` sits in the
    denominator, and the estimator itself is undefined there. Small ``a``
    is fine; ``a`` near 1 is not, and the routine refuses it rather than
    returning a huge number that looks like a result.

    Parameters
    ----------
    n : int
        Sample size.
    h : float
        Bandwidth.
    a : float
        Second smoothing parameter; ``a > 0``, ``a != 1``.
    fx : float
        ``F_X(x)``.
    density : float
        ``f_X(x)``.
    r1, r2 : float, optional
        The kernel constants; default to the Gaussian ``r_1`` and the
        ``r_2`` evaluated at this ``a``.

    Returns
    -------
    RichResult
        Keys ``variance``, ``se``, ``edfvar``, ``gain``, ``r1``, ``r2``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 2.3, Eqs. (2.9)-(2.10).
    """
    from .fzr1 import kdfr1
    from .fzr2 import kdfr2

    n = int(n)
    h = float(h)
    a = float(a)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}.")
    if abs(a - 1.0) < 1e-6:
        raise ValueError("a too close to 1: (a^2 - 1)^2 divides the variance.")
    if r1 is None:
        r1 = float(kdfr1()["estimate"])
    if r2 is None:
        r2 = float(kdfr2(a=a)["estimate"])
    bracket = 2.0 * (a ** 4 + 1.0) / (a * a - 1.0) ** 2 * float(r1) + float(r2)
    edfvar = float(fx) * (1.0 - float(fx)) / n
    var = edfvar - h / n * bracket * float(density)
    return RichResult(
        payload={
            "variance": float(var),
            "se": float(np.sqrt(var)) if var > 0 else float("nan"),
            "edfvar": float(edfvar),
            "gain": float(edfvar - var),
            "r1": float(r1),
            "r2": float(r2),
            "method": "variance of the bias-reduced KDFE (Theorem 2.3)",
        }
    )


fauzi_thm2_3_var_brdkdfe = gekdfvar


def cheatsheet():
    return "fzt23: same variance ORDER as the plain KDFE, smaller constant; blows up as a -> 1 (Thm 2.3)"


# CANONICAL TEST
# >>> r = gekdfvar(n=100, h=0.2, a=2.0, fx=0.5, density=0.4)
# >>> r['variance'] < r['edfvar']
# True
