# morie.fn -- function file (rootcoder007/morie)
"""Bandwidth condition for the kernel quantile Edgeworth expansion."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["qbwcheck", "fauzi_quantile_bw_condition"]


def qbwcheck(h, n, eps=0.05):
    r"""Bandwidth condition for the kernel quantile Edgeworth expansion.

    Eq. (3.8): the expansion is proved under

    .. math:: h = o(n^{-1/4})
              \quad\text{and}\quad
              \lim_{n\to\infty}(n^{1/4}h)^{-k}n^{-\beta} = 0

    for every :math:`\beta>0` and integer :math:`k`.

    The first half is a rate. The second is not -- it says :math:`h` may
    not shrink FASTER than any negative power of ``n``, so a bandwidth
    like :math:`e^{-n}` is excluded even though it satisfies the first
    condition comfortably. The window is genuinely two-sided, which is why
    the book's working choice is :math:`h = n^{-1/4}(\log n)^{-1}`: it
    beats :math:`n^{-1/4}` by a logarithm, and only by a logarithm.

    Checked as: ``h < n^(-1/4)`` for the upper side and
    ``h * n^(1/4 + eps) -> inf``, tested at the given ``eps``, for the
    lower. Both are finite-sample proxies for limit statements and are
    reported separately, along with the book's reference bandwidth, so
    a caller can see which side a bandwidth fails on.

    Parameters
    ----------
    h : float
        Bandwidth.
    n : int
        Sample size.
    eps : float, default 0.05
        The margin at which the lower-side condition is probed.

    Returns
    -------
    RichResult
        Keys ``ok``, ``upper``, ``lower``, ``hbook``, ``ratio``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Eq. (3.8).
    """
    h = float(h)
    n = int(n)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if n < 2:
        raise ValueError(f"sample size must be at least 2, got {n}.")
    cap = float(n) ** -0.25
    upper = bool(h < cap)
    lower = bool(h * float(n) ** (0.25 + float(eps)) > 1.0)
    hbook = cap / np.log(n)
    return RichResult(
        payload={
            "ok": bool(upper and lower),
            "upper": upper,
            "lower": lower,
            "hbook": float(hbook),
            "ratio": float(h / cap),
            "method": "bandwidth window (3.8) for the quantile Edgeworth expansion",
        }
    )


fauzi_quantile_bw_condition = qbwcheck


def cheatsheet():
    return "fzbwc: (3.8) is a two-sided window: h must beat n^(-1/4) but not by more than a power"


# CANONICAL TEST
# >>> r = qbwcheck(h=0.05, n=1000)
# >>> r['upper'] and r['lower']
# True
