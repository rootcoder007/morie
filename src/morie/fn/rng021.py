# morie.fn -- function file (rootcoder007/morie)
"""Covariance and correlation coefficient (Rangayyan eqs. 3.21-3.22)."""


from math import fsum, sqrt

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["covxy", "rangayyan_ch3_covariance"]


def covxy(x, y, ddof=0):
    """Covariance between two processes and the correlation it normalizes to.

    Rangayyan (2024) eqs. (3.21)-(3.22):
        C_xy  = E[(x - mu_x)(y - mu_y)]
        rho   = C_xy / (sigma_x sigma_y),   -1 <= rho <= +1.

    ``ddof=0`` matches the population divisor used throughout Section
    3.2.1; pass ``ddof=1`` for the unbiased sample covariance.  rho is
    None when either process is constant, since eq. (3.22) divides by a
    zero SD there rather than being zero.
    """
    xs, ys = aslist(x), aslist(y)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    n = len(xs)
    d = n - int(ddof)
    if d <= 0:
        raise ValueError("not enough samples for ddof=%d" % ddof)
    mx, my = fsum(xs) / n, fsum(ys) / n
    cov = fsum((a - mx) * (b - my) for a, b in zip(xs, ys)) / d
    vx = fsum((a - mx) ** 2 for a in xs) / d
    vy = fsum((b - my) ** 2 for b in ys) / d
    rho = cov / sqrt(vx * vy) if vx > 0 and vy > 0 else None
    return RichResult(payload={
        "covariance": cov, "correlation": rho, "sd_x": sqrt(vx),
        "sd_y": sqrt(vy), "mean_x": mx, "mean_y": my, "n": n, "ddof": int(ddof),
        "method": "Rangayyan (2024) eqs. (3.21)-(3.22)"})


rangayyan_ch3_covariance = covxy  # pre-policy spelling


def cheatsheet():
    return "rng021: covariance and correlation, Rangayyan eqs. (3.21)-(3.22)"
