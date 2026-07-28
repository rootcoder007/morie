# morie.fn -- function file (rootcoder007/morie)
"""Glivenko-Cantelli sup-distance diagnostic."""

import numpy as np

from ._richresult import RichResult

__all__ = ["gibbons_glivenko_cantelli"]


def gibbons_glivenko_cantelli(x, F=None):
    r"""Theorem 2.3.2 (Glivenko-Cantelli): the EDF converges to F
    *uniformly*, almost surely:

    .. math:: P\big(\lim_{n\to\infty} \sup_x |S_n(x) - F(x)| = 0\big)
              = 1.

    A theorem has no finite-sample output, so this returns the finite
    -sample witness: the observed sup distance
    :math:`D_n = \sup_x |S_n(x) - F(x)|` together with the
    Dvoretzky-Kiefer-Wolfowitz bound
    :math:`P(D_n > \epsilon) \le 2 e^{-2n\epsilon^2}`, which is the
    quantitative content behind the almost-sure statement.

    Parameters
    ----------
    x : array-like
        Sample.
    F : callable, optional
        True CDF; standard normal if omitted.

    Returns
    -------
    RichResult
        keys: ``sup_distance``, ``dkw_bound_at_observed``
        (2 exp(-2 n D_n^2)), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 2.3.2.
    """
    from scipy import stats

    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n < 1:
        raise ValueError("x must be non-empty.")
    Fv = stats.norm.cdf(x) if F is None else np.asarray([F(v) for v in x], dtype=float)
    up = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    D = float(np.max(np.maximum(np.abs(up - Fv), np.abs(Fv - lo))))
    return RichResult(
        payload={
            "sup_distance": D,
            "dkw_bound_at_observed": float(min(1.0, 2.0 * np.exp(-2.0 * n * D**2))),
            "n": int(n),
            "method": "sup |S_n - F| with the DKW bound (Gibbons Theorem 2.3.2)",
        }
    )


def cheatsheet():
    return "gb232: D_n witness + DKW 2exp(-2nD^2); the quantitative GC"
