# morie.fn -- function file (rootcoder007/morie)
"""Large-sample moments of order statistics."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_large_sample_moments"]


def gibbons_large_sample_moments(r, n, f=None, F_inv=None, f_prime=None):
    r"""Section 2.9: David-Johnson approximations for the rth order
    statistic with p = r/(n + 1), q = 1 - p:

    .. math::

       E(X_{(r)}) &\approx F^{-1}(p)
         + \frac{p q}{2(n + 2)} \cdot \frac{-f'(F^{-1}(p))}{f^3} \\
       \mathrm{Var}(X_{(r)}) &\approx \frac{p q}{(n + 2)\,
         f(F^{-1}(p))^2}

    The second-order mean term uses Q''(p) = -f'/f^3 from Theorem
    2.2.1 -- the two modules share the same derivative identity.

    Parameters
    ----------
    r : int
        Order-statistic index, 1 <= r <= n.
    n : int
        Sample size.
    f : callable, optional
        Parent density; standard normal if omitted.
    F_inv : callable, optional
        Parent quantile function; standard normal if omitted.
    f_prime : callable, optional
        Density derivative; numerically differenced if omitted.

    Returns
    -------
    RichResult
        keys: ``mean``, ``var``, ``sd``, ``p``, ``first_order_mean``
        (plain F^{-1}(p)), ``r``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 2.9.

    David, H. A. & Nagaraja, H. N. (2003). *Order Statistics*
    (3rd ed.). Wiley. Sec. 4.6.
    """
    from scipy import stats

    r, n = int(r), int(n)
    if not 1 <= r <= n:
        raise ValueError(f"need 1 <= r <= n, got r={r}, n={n}.")
    if f is None:
        f = stats.norm.pdf
    if F_inv is None:
        F_inv = stats.norm.ppf
    p = r / (n + 1.0)
    q = 1.0 - p
    xq = float(F_inv(p))
    fq = float(f(xq))
    if fq <= 0:
        raise ValueError("density is zero at the quantile.")
    h = 1e-6
    fpq = (float(f(xq + h)) - float(f(xq - h))) / (2 * h) if f_prime is None \
        else float(f_prime(xq))
    mean = xq + p * q / (2.0 * (n + 2)) * (-fpq / fq**3)
    var = p * q / ((n + 2.0) * fq**2)
    return RichResult(
        payload={
            "mean": float(mean), "var": float(var), "sd": float(np.sqrt(var)),
            "p": p, "first_order_mean": xq, "r": r, "n": n,
            "method": "David-Johnson large-sample moments (Gibbons Ch. 2.9)",
        }
    )


def cheatsheet():
    return "gb_lsm: E ~ Q(p) + pq Q''(p)/(2(n+2)); Var ~ pq/((n+2) f^2)"
