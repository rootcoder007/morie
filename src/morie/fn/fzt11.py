# morie.fn -- function file (rootcoder007/morie)
"""Bias and variance of the raw gamma-kernel function A_h (Theorem 1.1)."""

from . import _array_core as np
from ._fauzi import rratio
from ._richresult import RichResult

__all__ = ["gkrawbv", "fauzi_thm1_1_bias_mgkde"]


def gkrawbv(x, h, n, fp, fpp, f, boundary=False, c=None):
    r"""Bias (1.10) and variance (1.11) of ``A_h(x)``, Eq. (1.9).

    ``A_h`` is the raw gamma-kernel function -- the book is explicit in
    Remark 1.1 that it is NOT the proposed estimator, only the object the
    proposed one is extrapolated from. Its moments are

    .. math::
        \mathrm{Bias}[A_h(x)] &= \Big(f_X'(x) + \tfrac12 x^2 f_X''(x)\Big)
            \sqrt h + o(\sqrt h), \\
        \mathrm{Var}[A_h(x)] &= \frac{R^2(h^{-1/2}-1)\,f_X(x)}
            {2(x+\sqrt h)\sqrt\pi\,(1-\sqrt h)\,R(2h^{-1/2}-2)\,n h^{1/4}}

    in the interior (:math:`x/h \to \infty`), and the same expression with
    :math:`(x+\sqrt h)` replaced by :math:`(c\sqrt h + 1)` and
    :math:`h^{1/4}` by :math:`h^{3/4}` in the boundary region
    (:math:`x/h \to c`). ``R`` is Eq. (1.12).

    The bias is O(sqrt h) -- WORSE than Chen's O(h). That is the whole
    tension of Sec. 1.2: fixing the gamma shape at :math:`h^{-1/2}` and
    moving the scale buys a variance of order :math:`n^{-1}h^{-1/4}`
    instead of Chen's :math:`n^{-1}h^{-1/2}`, and pays for it in bias.
    Theorem 1.2 buys the bias back by geometric extrapolation.

    Nothing here is estimated from data: the caller supplies the true (or
    pilot) ``f``, ``fp``, ``fpp`` at ``x``, and the routine returns the
    book's asymptotic expressions exactly. That keeps it deterministic and
    makes it usable as the reference in a simulation study, which is what
    Sec. 1.3 does.

    Parameters
    ----------
    x : float
        Evaluation point, ``x >= 0``.
    h : float
        Bandwidth, ``h > 0``.
    n : int
        Sample size.
    fp, fpp, f : float
        ``f_X'(x)``, ``f_X''(x)`` and ``f_X(x)``.
    boundary : bool, default False
        Use the boundary-region branch of (1.11).
    c : float, optional
        The constant in ``x/h -> c``; required when ``boundary`` is True.

    Returns
    -------
    RichResult
        Keys ``bias``, ``variance``, ``mse``, ``rnum``, ``rden``,
        ``region``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 1.1, Eqs. (1.9)-(1.12).
    """
    x = float(x)
    h = float(h)
    n = int(n)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if x < 0:
        raise ValueError("gamma kernels need x >= 0.")
    rh = np.sqrt(h)
    bias = (float(fp) + 0.5 * x * x * float(fpp)) * rh
    rnum = float(np.atleast_1d(rratio(1.0 / rh - 1.0))[0])
    rden = float(np.atleast_1d(rratio(2.0 / rh - 2.0))[0])
    if boundary:
        if c is None:
            raise ValueError("the boundary branch of (1.11) needs c.")
        scale = float(c) * rh + 1.0
        power = h ** 0.75
        region = "boundary"
    else:
        scale = x + rh
        power = h ** 0.25
        region = "interior"
    var = (rnum ** 2 * float(f)) / (2.0 * scale * np.sqrt(np.pi) * (1.0 - rh) * rden * n * power)
    return RichResult(
        payload={
            "bias": float(bias),
            "variance": float(var),
            "mse": float(bias * bias + var),
            "rnum": rnum,
            "rden": rden,
            "region": region,
            "h": h,
            "n": n,
            "method": "gamma-kernel A_h bias and variance (Theorem 1.1)",
        }
    )


# legacy backlog spelling; the label it carried ("bias of modified gamma
# KDE is O(h^4)") is wrong on two counts -- Theorem 1.1 is about A_h, not
# the modified estimator, and its bias is O(sqrt h), not O(h^4).
fauzi_thm1_1_bias_mgkde = gkrawbv


def cheatsheet():
    return "fzt11: bias O(sqrt h) and variance of the raw gamma-kernel A_h (Thm 1.1)"


# CANONICAL TEST
# >>> r = gkrawbv(x=1.0, h=0.01, n=100, fp=0.1, fpp=-0.2, f=0.3)
# >>> abs(r["bias"] - (0.1 - 0.1) * 0.1) < 1e-15
# True
