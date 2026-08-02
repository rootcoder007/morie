# morie.fn -- function file (rootcoder007/morie)
"""Distribution-free tolerance intervals from order statistics."""

from . import _array_core as np
from scipy import special, stats

from ._richresult import RichResult

__all__ = ["gibbons_tolerance_beta"]


def gibbons_tolerance_beta(x=None, r=1, s=None, p=0.9, gamma=None, n=None):
    r"""Theorem 2.11.1 (PDF-verified, printed p. 61): for a sample of
    size n from ANY continuous distribution, the coverage of the
    interval :math:`(X_{(r)}, X_{(s)})` is

    .. math:: U_{(s)} - U_{(r)} \sim \mathrm{Beta}(s - r,\;
              n - s + r + 1),

    so the tolerance coefficient is

    .. math:: \gamma = P(\text{coverage} \ge p)
              = 1 - I_p(s - r,\, n - s + r + 1).

    Distribution-free: only continuity of F is used, via the
    probability integral transformation.

    Parameters
    ----------
    x : array-like, optional
        Sample; when given, the interval endpoints are reported and n
        is taken from it.
    r, s : int
        Order-statistic indices, 1 <= r < s <= n; s defaults to n.
    p : float in (0, 1), default 0.9
        Required coverage.
    gamma : float in (0, 1), optional
        When given INSTEAD of being computed, the smallest n with
        (r, s) = (1, n) achieving this tolerance coefficient is found.
    n : int, optional
        Sample size when x is omitted.

    Returns
    -------
    RichResult
        keys: ``gamma`` (achieved tolerance coefficient),
        ``coverage_dist`` ((a, b) of the Beta), ``endpoints`` (if x
        given), ``n_required`` (if gamma given), ``r``, ``s``, ``p``,
        ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.11.1.
    """
    if not 0 < p < 1:
        raise ValueError(f"p must lie in (0, 1), got {p}.")
    endpoints = None
    if x is not None:
        x = np.sort(np.asarray(x, dtype=float).ravel())
        n = x.size
    if n is None and gamma is None:
        raise ValueError("supply x, n, or gamma.")

    if gamma is not None and n is None:
        if not 0 < gamma < 1:
            raise ValueError(f"gamma must lie in (0, 1), got {gamma}.")
        # smallest n with P(U(n) - U(1) >= p) >= gamma:
        # coverage ~ Beta(n - 1, 2)
        for m in range(2, 100000):
            if 1.0 - special.betainc(m - 1, 2, p) >= gamma:
                return RichResult(
                    payload={
                        "n_required": int(m), "gamma": float(gamma), "p": float(p),
                        "r": 1, "s": m, "n": None, "coverage_dist": (m - 1, 2),
                        "endpoints": None,
                        "method": "Smallest n for (X(1), X(n)) tolerance (Thm 2.11.1)",
                    }
                )
        raise ValueError("no n below 100000 achieves that tolerance.")

    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    s = n if s is None else int(s)
    r = int(r)
    if not 1 <= r < s <= n:
        raise ValueError(f"need 1 <= r < s <= n, got r={r}, s={s}, n={n}.")
    a, b = s - r, n - s + r + 1
    g = 1.0 - special.betainc(a, b, p)
    if x is not None:
        endpoints = (float(x[r - 1]), float(x[s - 1]))
    return RichResult(
        payload={
            "gamma": float(g), "coverage_dist": (a, b), "endpoints": endpoints,
            "n_required": None, "r": r, "s": s, "p": float(p), "n": n,
            "method": "Coverage ~ Beta(s-r, n-s+r+1) (Gibbons Theorem 2.11.1)",
        }
    )


def cheatsheet():
    return "gb2111: gamma = 1 - I_p(s-r, n-s+r+1); distribution-free via PIT"
