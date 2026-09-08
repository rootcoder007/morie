# morie.fn -- function file (rootcoder007/morie)
"""Birnbaum-Tingey exact one-sided K-S tail."""

from math import comb, floor

from ._richresult import RichResult

__all__ = ["gibbons_ks_bt_formula"]


def gibbons_ks_bt_formula(c, n):
    r"""Birnbaum-Tingey closed form (Gibbons eq. 4.3.5) for the exact
    one-sided tail:

    .. math:: P(D_n^+ > c) = (1 - c)^n + c \sum_{j=1}^{\lfloor
              n(1-c)\rfloor} \binom{n}{j} \Big(1 - c -
              \frac{j}{n}\Big)^{n-j} \Big(c + \frac{j}{n}\Big)^{j-1}

    Exact at every n -- no asymptotics -- which is what makes it the
    reference against which the exponential limit (Theorem 4.3.5) is
    checked.

    Parameters
    ----------
    c : float in (0, 1)
        Threshold.
    n : int
        Sample size.

    Returns
    -------
    RichResult
        keys: ``p_exceed``, ``cdf`` (1 - p), ``c``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Eq. (4.3.5).

    Birnbaum, Z. W. & Tingey, F. H. (1951). One-sided confidence
    contours for probability distribution functions. *The Annals of
    Mathematical Statistics*, 22(4), 592-596.
    """
    c = float(c)
    if not 0 < c < 1:
        raise ValueError(f"c must lie in (0, 1), got {c}.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    p = (1.0 - c) ** n
    for j in range(1, floor(n * (1.0 - c)) + 1):
        p += c * comb(n, j) * (1.0 - c - j / n) ** (n - j) * (c + j / n) ** (j - 1)
    p = min(max(p, 0.0), 1.0)
    return RichResult(
        payload={
            "p_exceed": float(p), "cdf": float(1.0 - p), "c": c, "n": n,
            "method": "Birnbaum-Tingey exact P(D+_n > c) (Gibbons eq. 4.3.5)",
        }
    )


def cheatsheet():
    return "gb434bt: exact one-sided K-S tail, no asymptotics"
