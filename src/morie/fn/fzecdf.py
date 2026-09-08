# morie.fn -- function file (rootcoder007/morie)
"""Empirical distribution function."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["edf", "fauzi_ecdf"]


def edf(x, grid=None):
    r"""Empirical distribution function.

    .. math:: F_n(x) = \frac1n\sum_{i=1}^n I(X_i \le x).

    The book's baseline, and the thing every kernel estimator in it is
    trying to beat. Its bias is exactly zero and its variance is exactly
    :math:`F(x)(1-F(x))/n` -- no expansion, no remainder -- which is why
    Sec. 2.1 can say flatly that a kernel with :math:`r_1 > 0` beats it
    for ANY :math:`F_X`: the kernel estimator's variance is that same
    quantity minus :math:`2hr_1f_X(x)/n`.

    Right-continuous, as the definition requires: ties at ``x`` count.
    ``se`` is the exact binomial standard error, not an asymptotic one.

    Parameters
    ----------
    x : array-like
        Sample.
    grid : array-like, optional
        Evaluation points; defaults to the sorted sample.

    Returns
    -------
    RichResult
        Keys ``estimate``, ``se``, ``variance``, ``grid``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 2.1, and the bias/variance display preceding (2.3).
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 1:
        raise ValueError("need at least one observation.")
    g = np.sort(xv) if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    est = np.asarray([float(np.mean(xv <= float(t))) for t in g], dtype=float)
    var = est * (1.0 - est) / n
    return RichResult(
        payload={
            "estimate": [float(v) for v in est],
            "se": [float(v) for v in np.sqrt(var)],
            "variance": [float(v) for v in var],
            "grid": [float(v) for v in g],
            "n": int(n),
            "method": "empirical distribution function",
        }
    )


fauzi_ecdf = edf


def cheatsheet():
    return "fzecdf: empirical df -- zero bias, variance F(1-F)/n exactly; the baseline to beat"


# CANONICAL TEST
# >>> r = edf([1.0, 2.0, 3.0, 4.0], grid=[2.0])
# >>> r['estimate'][0] == 0.5
# True
