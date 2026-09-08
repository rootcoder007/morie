# morie.fn -- function file (rootcoder007/morie)
"""Empirical distribution function."""

from . import _array_core as np

from ._kosorok import empirical_df
from ._richresult import RichResult

__all__ = ["kosorok_ch2_empirical_distribution_function"]


def kosorok_ch2_empirical_distribution_function(X, t=None, n=None):
    r"""Empirical distribution function

    .. math:: F_n(t) = n^{-1}\sum_{i=1}^{n} 1\{X_i \le t\}.

    The base object of the whole chapter: every process below is a
    functional of this one. Evaluated on the sample's own order
    statistics when ``t`` is omitted, since those are the only points
    where F_n changes.

    Parameters
    ----------
    X : array-like
        Sample.
    t : float or array-like, optional
        Evaluation points; the sorted sample if omitted.
    n : int, optional
        Accepted for interface compatibility; taken from X.

    Returns
    -------
    RichResult
        keys: ``t``, ``F_n``, ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2.
    """
    X = np.asarray(X, dtype=float).ravel()
    if X.size < 1:
        raise ValueError("X must be non-empty.")
    if n is not None and int(n) != X.size:
        raise ValueError(f"n = {n} does not match len(X) = {X.size}.")
    tt = np.sort(X) if t is None else np.atleast_1d(np.asarray(t, dtype=float))
    return RichResult(
        payload={"t": tt, "F_n": empirical_df(X, tt), "n": int(X.size),
                 "method": "F_n(t) = n^-1 sum 1{X_i <= t} (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr026: the EDF; base object of the chapter"
