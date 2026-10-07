# morie.fn -- function file (rootcoder007/morie)
"""Spearman rank correlation coefficient (rho)."""

import functools
import itertools
import math
from typing import Union

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

# Edgeworth-series coefficients of AS 89 (Best & Roberts 1975), as R's prho.c
_AS89 = (0.2274, 0.2531, 0.1745, 0.0758, 0.1033, 0.3932, 0.0879, 0.0151, 0.0072, 0.0831, 0.0131, 4.6e-4)


@functools.cache
def _s_values(n: int) -> tuple:
    """S = sum (r_i - i)^2 over all n! permutations (n <= 9), for the exact tail."""
    return tuple(sum((i + 1 - r) ** 2 for i, r in enumerate(perm)) for perm in itertools.permutations(range(1, n + 1)))


def _prho(n: int, s: float, lower: bool) -> float:
    """AS 89 as R's prho.c: P[S >= s] (upper) or P[S < s] (lower); exact for n <= 9,
    an Edgeworth series beyond."""
    pv = 0.0 if lower else 1.0
    if n <= 1 or s <= 0:
        return pv
    n3 = n * (n * n - 1) / 3
    if s > n3:
        return 1 - pv
    if n <= 9:
        nfac = math.factorial(n)
        ifr = 1 if s == n3 else sum(1 for v in _s_values(n) if s <= v)
        return (nfac - ifr if lower else ifr) / nfac
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = _AS89
    b = 1 / n
    x = (6 * (s - 1) * b / (n * n - 1) - 1) * math.sqrt(1 / b - 1)
    y = x * x
    u = (
        x
        * b
        * (
            c1
            + b * (c2 + c3 * b)
            + y * (-c4 + b * (c5 + c6 * b) - y * b * (c7 + c8 * b - y * (c9 - c10 * b + y * b * (c11 - c12 * y))))
        )
    )
    y = u / math.exp(y / 2)
    pv = (-y if lower else y) + 0.5 * math.erfc((-x if lower else x) / math.sqrt(2))
    return min(1.0, max(0.0, pv))


def _spearman_exact_p(rho: float, n: int) -> float:
    """cor.test(method = "spearman", exact = TRUE)'s two-sided p-value."""
    q = (n**3 - n) * (1 - rho) / 6
    lower = not q > (n**3 - n) / 6
    p = _prho(n, round(q) + 2 * lower, lower)
    return min(2 * p, 1.0)


def spearman_rho(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
) -> dict:
    """
    Spearman rank correlation coefficient (rho).

    Pearson correlation applied to ranks; captures monotone (not just linear)
    associations and is robust to outliers.

    :param x: First variable (1-D array-like).
    :param y: Second variable (same length as x).
    :return: dict with keys ``rho``, ``p_value``.
    :raises ValueError: If x and y have different lengths or fewer than 3 observations.

    The p-value is R's ``cor.test(method = "spearman")`` default: algorithm AS 89 (exact
    for n <= 9, an Edgeworth series up to n < 1290) for untied data, the t approximation
    with ties or larger n -- the same as rmorie's ``morie_spearman_rho``.

    Examples
    --------
    >>> x = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8]
    >>> z = [6.1, 5.2, 6.9, 7.4, 6.0, 5.8, 7.1, 6.6, 6.2, 6.4]
    >>> round(spearman_rho(x, z)["p_value"], 12)
    0.864753528805

    References
    ----------
    Spearman, C. (1904). The proof and measurement of association between two things.
        American Journal of Psychology, 15(1), 72-101.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if len(ax) != len(ay):
        raise ValueError("x and y must have the same length.")
    if len(ax) < 3:
        raise ValueError("At least 3 observations are required.")
    rho, p_val = stats.spearmanr(ax, ay)
    # cor.test's default, as rmorie's morie_spearman_rho: the exact / Edgeworth (AS 89) p-value
    # for untied data with n < 1290, the t approximation otherwise
    n = len(ax)
    if n < 1290 and len(set(ax.tolist())) == n and len(set(ay.tolist())) == n:
        p_val = _spearman_exact_p(float(rho), n)
    return RichResult(payload={"rho": float(rho), "p_value": float(p_val)})


rho = spearman_rho


def cheatsheet() -> str:
    return "spearman_rho({}) -> Spearman rank correlation coefficient (rho)."


# compact alias per ledger/NAMING.md
spearmanrho = spearman_rho
