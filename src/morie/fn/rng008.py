# morie.fn -- function file (rootcoder007/morie)
"""Sample mean square."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sample_mean_squared"]


def rangayyan_ch3_sample_mean_squared(eta, N=None):
    r"""Sample mean-squared value (Rangayyan Ch. 3):

    .. math:: MS_\eta = \frac1N \sum_{n=0}^{N-1} [\eta(n)]^2.

    This is the total average power, NOT the variance: the two differ
    by the squared mean, and they coincide only for a zero-mean
    signal. Both are returned so the distinction is visible.

    Parameters
    ----------
    eta : array-like
        Samples.
    N : int, optional
        Length.

    Returns
    -------
    RichResult
        keys: ``mean_square``, ``variance``, ``mean``, ``N``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    if eta.size < 1:
        raise ValueError("eta must be non-empty.")
    if N is not None and int(N) != eta.size:
        raise ValueError(f"N = {N} does not match len(eta) = {eta.size}.")
    mu = float(np.mean(eta))
    ms = float(np.mean(eta**2))
    return RichResult(payload={"mean_square": ms, "variance": ms - mu**2,
                               "mean": mu, "N": int(eta.size),
                               "method": "MS = (1/N) sum eta^2; equals variance only if mu = 0"})


def cheatsheet():
    return "rng008: mean square is total power, not variance, unless mu = 0"
