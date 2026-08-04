# morie.fn -- function file (rootcoder007/morie)
"""Bias and variance of the boundary-free KDFE (Theorem 5.2)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkdfbv", "fauzi_thm5_2_bdfree_kdfe_bv"]


def bfkdfbv(n, h, fx, density, c1, dg, mu2=1.0, r1=None):
    r"""Bias and variance of the boundary-free KDFE (Theorem 5.2).

    Theorem 5.2, Eqs. (5.6)-(5.7):

    .. math::
        \mathrm{Bias}[\tilde F_X(x)] &= \frac{h^2}2 c_1(x)\mu_2(K)
            + o(h^2), \\
        \mathrm{Var}[\tilde F_X(x)] &= \frac{F_X(1-F_X)}n
            - \frac{2h}n g'(g^{-1}(x))f_X(x)\,r_1 + o\!\Big(\frac hn\Big),

    with :math:`c_1` from (5.8) and :math:`r_1` from (2.9).

    Remark 5.1 draws the consequence and it is worth stating exactly:
    since :math:`r_1 > 0` and :math:`g` is increasing, the variance is
    SMALLER than the naive estimator's whenever
    :math:`g'(g^{-1}(x)) \ge 1`. The bias comparison is not settled in
    general, but for :math:`\Omega = \mathbb R^+` with :math:`g = \exp`
    the bias converges faster in the boundary region as
    :math:`x \to 0` -- which is the case the construction exists for.

    So ``vargain`` is returned explicitly: it is
    :math:`2hg'f_Xr_1/n`, the amount by which smoothing beats the
    empirical df here, and it is positive exactly when the transformation
    stretches.

    Parameters
    ----------
    n : int
        Sample size.
    h : float
        Bandwidth.
    fx : float
        ``F_X(x)``.
    density : float
        ``f_X(x)``.
    c1 : float
        The coefficient (5.8).
    dg : float
        ``g'(g^{-1}(x))``.
    mu2 : float, default 1.0
        ``int v^2 K(v) dv``.
    r1 : float, optional
        Kernel constant (2.9); defaults to the Gaussian value.

    Returns
    -------
    RichResult
        Keys ``bias``, ``variance``, ``se``, ``edfvar``, ``vargain``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.2, Eqs. (5.6)-(5.8).
    """
    from .fzr1 import kdfr1

    n = int(n)
    h = float(h)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if r1 is None:
        r1 = float(kdfr1()["estimate"])
    bias = (h * h / 2.0) * float(c1) * float(mu2)
    edfvar = float(fx) * (1.0 - float(fx)) / n
    gain = 2.0 * h / n * float(dg) * float(density) * float(r1)
    var = edfvar - gain
    return RichResult(
        payload={
            "bias": float(bias),
            "variance": float(var),
            "se": float(np.sqrt(var)) if var > 0 else float("nan"),
            "edfvar": float(edfvar),
            "vargain": float(gain),
            "h": h,
            "n": n,
            "method": "boundary-free KDFE bias and variance (Theorem 5.2)",
        }
    )


fauzi_thm5_2_bdfree_kdfe_bv = bfkdfbv


def cheatsheet():
    return "fzt52: Thm 5.2: variance beats the naive one whenever g' >= 1 (Remark 5.1)"


# CANONICAL TEST
# >>> r = bfkdfbv(n=100, h=0.2, fx=0.5, density=0.4, c1=0.3, dg=1.0)
# >>> r['variance'] < r['edfvar']
# True
