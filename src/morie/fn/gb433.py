# morie.fn -- function file (rootcoder007/morie)
"""Kolmogorov limiting distribution of D_n."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_ks_kolmogorov_limit"]


def gibbons_ks_kolmogorov_limit(d, n=None):
    r"""Theorem 4.3.3 (PDF-verified, printed p. 108): for any
    continuous F,

    .. math:: \lim_{n\to\infty} P(D_n \le d/\sqrt n) = L(d)
              = 1 - 2\sum_{i=1}^{\infty} (-1)^{i-1} e^{-2 i^2 d^2}.

    The series is summed to machine convergence. When n is supplied,
    the approximate P(D_n <= d) at that n is also returned by
    evaluating L at d sqrt(n).

    Parameters
    ----------
    d : float > 0
        The argument of L (or the raw statistic when n is given).
    n : int, optional
        Sample size for the finite-n approximation.

    Returns
    -------
    RichResult
        keys: ``L``, ``p_value`` (1 - L, the upper tail), ``terms``
        (series terms used), ``finite_n_cdf`` (if n given), ``d``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 4.3.3.

    Kolmogorov, A. N. (1933). Sulla determinazione empirica di una
    legge di distribuzione. *Giornale dell'Istituto Italiano degli
    Attuari*, 4, 83-91.
    """
    d = float(d)
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")

    def L(t):
        s = 0.0
        i = 1
        while True:
            term = (-1) ** (i - 1) * np.exp(-2.0 * i**2 * t**2)
            s += term
            if abs(term) < 1e-16 or i > 200:
                break
            i += 1
        return 1.0 - 2.0 * s, i

    Ld, terms = L(d)
    Ld = float(min(max(Ld, 0.0), 1.0))
    payload = {
        "L": Ld, "p_value": float(1.0 - Ld), "terms": int(terms), "d": d,
        "method": "L(d) = 1 - 2 sum (-1)^{i-1} exp(-2 i^2 d^2) (Theorem 4.3.3)",
    }
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        payload["finite_n_cdf"] = float(min(max(L(d * np.sqrt(n))[0], 0.0), 1.0))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb433: Kolmogorov L(d); p = 1 - L"
