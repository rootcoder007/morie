# morie.fn -- function file (rootcoder007/morie)
"""Higher-order kernel for bias reduction (Fauzi Ch 1).

A kernel K_r is of order r if

    integral K_r(u) du     = 1,
    integral u^j K_r(u) du = 0   for j = 1, …, r-1,
    integral u^r K_r(u) du != 0.

Higher-order kernels reduce the leading-order bias of a KDE from
O(h^2) (order-2) to O(h^r), at the cost of allowing K_r(u) < 0 (so the
density estimate is not guaranteed non-negative).

The Gaussian-based kernels of every even order ``2r`` are

    K_2r(u) = phi(u) * sum_{j=0}^{r-1} (-1)^j He_2j(u) / (2^j j!)

(He the probabilists' Hermite polynomials; Wand & Jones 1995, sec. 2.8):
order 2 is phi itself and order 4 the textbook K_4(u) = (1/2)(3 - u^2)
phi(u), with fourth moment -3.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["fauzi_higher_order_kernel"]


def _silverman_h(x):
    n = len(x)
    s = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    sigma = min(s, iqr) if iqr > 0 else s
    if sigma <= 0:
        sigma = 1.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def _poly(order):
    """Coefficients c_m of the polynomial P with K(u) = P(u) phi(u)."""
    if order < 2 or order % 2:
        raise ValueError("order must be an even integer >= 2")
    c = [0.0] * order
    for j in range(order // 2):
        # He_2j(u) = sum_k (-1)^k (2j)! / (k! (2j - 2k)! 2^k) u^(2j - 2k)
        s = (-1) ** j / (2**j * math.factorial(j))
        for k in range(j + 1):
            c[2 * j - 2 * k] += (
                s * (-1) ** k * math.factorial(2 * j) / (math.factorial(k) * math.factorial(2 * j - 2 * k) * 2**k)
            )
    return c


def _moments(c, order):
    """mu_order = int u^order K and R(K) = int K^2, from int u^m phi = (m-1)!! and int u^m phi^2 = Gamma((m+1)/2)/(2 pi)."""

    def dfact(m):
        return 1.0 if m <= 0 else float(math.prod(range(m - 1, 0, -2)))

    mu = sum(cm * dfact(order + m) for m, cm in enumerate(c) if (order + m) % 2 == 0)
    rk = sum(
        c[a] * c[b] * math.gamma((a + b + 1) / 2.0) / (2.0 * math.pi)
        for a in range(len(c))
        for b in range(len(c))
        if (a + b) % 2 == 0
    )
    return mu, rk


def fauzi_higher_order_kernel(x, t=None, h=None, order=4):
    """Kernel density estimate at ``t`` with the Gaussian-based kernel of even ``order``.

    ``f(t) = (1/(n h)) sum_i K_order((t - x_i)/h)``; ``h`` defaults to
    Silverman's rule. ``mu_r`` is the kernel's ``order``-th moment and
    ``R_K`` its roughness ``int K^2`` (both exact).

    Examples
    --------
    >>> r = fauzi_higher_order_kernel([0.1, -0.4, 0.3, 1.2, -0.8, 0.5], t=0.0, h=0.6)
    >>> round(r["estimate"], 12), r["mu_r"]
    (0.520884211539, -3.0)
    >>> round(fauzi_higher_order_kernel([0.1, -0.4, 0.3, 1.2, -0.8, 0.5], order=6)["mu_r"], 12)
    15.0
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return RichResult(payload={"estimate": np.nan, "n": n, "method": "fzhok -- too few obs"})
    c = _poly(int(order))
    if t is None:
        t = float(np.median(x))
    if h is None:
        h = float(_silverman_h(x))

    u = [(t - v) / h for v in x.tolist()]
    k = [sum(cm * ui**m for m, cm in enumerate(c)) * math.exp(-0.5 * ui * ui) / math.sqrt(2.0 * math.pi) for ui in u]
    f_hat = math.fsum(k) / (n * h)
    mu4_K4, R_K4 = _moments(c, int(order))

    return RichResult(
        payload={
            "estimate": f_hat,
            "h": h,
            "t": t,
            "order": order,
            "mu_r": mu4_K4,
            "R_K": R_K4,
            "n": n,
            "method": f"Fauzi higher-order ({order}) Gaussian-based kernel density (Ch 1)",
        }
    )


def cheatsheet():
    return "fzhok: KDE with the Gaussian-based kernel of even order r (Wand and Jones 1995)"


# CANONICAL TEST
# >>> import numpy as np
# >>> rng = np.random.default_rng(0)
# >>> x = rng.standard_normal(2000)
# >>> r = fauzi_higher_order_kernel(x, t=0.0)
# >>> abs(r["estimate"] - 0.3989) < 0.1  # phi(0) = 1/sqrt(2pi) ≈ 0.3989
# True
