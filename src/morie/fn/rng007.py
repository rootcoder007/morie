# morie.fn -- function file (rootcoder007/morie)
"""Sample mean."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sample_mean"]


def rangayyan_ch3_sample_mean(eta, N=None):
    r"""Sample mean of a noise process (Rangayyan Ch. 3):

    .. math:: \mu_\eta = \frac1N \sum_{n=0}^{N-1} \eta(n).

    Parameters
    ----------
    eta : array-like
        Signal or noise samples.
    N : int, optional
        Length; taken from the data.

    Returns
    -------
    RichResult
        keys: ``mean``, ``N``, ``method``.
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
    return RichResult(payload={"mean": float(np.mean(eta)), "N": int(eta.size),
                               "method": "mu_eta = (1/N) sum eta(n)"})


def cheatsheet():
    return "rng007: sample mean of the noise process"
