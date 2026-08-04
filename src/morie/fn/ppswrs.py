# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hansen-Hurwitz estimator for sampling with unequal probabilities.

Source consulted: Hansen, M.H. & Hurwitz, W.N. (1943). On the theory of
sampling from finite populations.  *Annals of Mathematical Statistics* 14(4),
333-362 (Project Euclid 10.1214/aoms/1177731356).  A sample of size ``n`` is
drawn with replacement, unit ``i`` selected with probability ``p_i`` on each
draw.  The unbiased estimator of the population total and its variance are

    That = (1/n) sum_k y_k / p_k
    V(That) = (1/n) sum_i p_i (Y_i/p_i - T)^2

with the customary unbiased sample estimate of the variance

    vhat(That) = (1 / (n (n - 1))) sum_k (y_k/p_k - That)^2 .

When ``p_i`` is proportional to a size measure this is probability
proportional to size (pps) sampling with replacement.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pps_with_replacement"]


def pps_with_replacement(y, p, sizes=None):
    """Hansen-Hurwitz total, variance and standard error.

    Parameters
    ----------
    y : array-like
        Values observed on the ``n`` selected draws.
    p : array-like
        Per-draw selection probability of the unit obtained on each draw.
        Ignored when ``sizes`` is given.
    sizes : array-like, optional
        Population size measure; ``p`` is then taken as ``sizes / sum(sizes)``
        restricted to the drawn units.

    Returns
    -------
    RichResult
        estimate (total), se, variance, zbar, n, method.

    References
    ----------
    Hansen & Hurwitz (1943), Ann. Math. Statist. 14(4), 333-362.
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    if sizes is not None:
        s = np.atleast_1d(np.asarray(sizes, dtype=float)).ravel()
        pp = s / float(np.sum(s))
    else:
        pp = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    n = int(min(yy.size, pp.size))
    z = np.asarray([float(yy[i]) / float(pp[i]) for i in range(n)], dtype=float)
    est = float(np.sum(z)) / n
    if n > 1:
        var = float(np.sum((z - est) * (z - est))) / (n * (n - 1))
    else:
        var = float("nan")
    se = float(np.sqrt(var)) if var == var and var >= 0.0 else float("nan")
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "variance": var,
            "zbar": est,
            "z": z,
            "n": n,
            "method": "Hansen-Hurwitz pps-with-replacement estimator (Hansen & Hurwitz 1943)",
        }
    )


# CANONICAL TEST
# >>> # y_k proportional to p_k makes every y/p equal: zero variance
# >>> r = pps_with_replacement([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
# >>> assert abs(r["estimate"] - 10.0) < 1e-12
# >>> assert abs(r["variance"]) < 1e-20


def cheatsheet():
    return "ppswrs(y, p): Hansen-Hurwitz pps-with-replacement total + se."
