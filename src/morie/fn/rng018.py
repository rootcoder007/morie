# morie.fn -- function file (rootcoder007/morie)
"""Ensemble average."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ensemble_average_function"]


def rangayyan_ch3_ensemble_average_function(x_k, M=None):
    r"""Ensemble average across realisations (Rangayyan Ch. 3):

    .. math:: \bar x(t) = \mu_x(t) = \frac1M \sum_{k=1}^{M} x_k(t).

    Averaging ACROSS realisations at each fixed time, not along one
    record. For a non-stationary process these differ: the ensemble
    mean is a function of time, whereas a time average collapses to a
    single number. Coherent averaging of M repeats improves SNR by
    sqrt(M), which is returned.

    Parameters
    ----------
    x_k : array-like, shape (M, T)
        One realisation per row.
    M : int, optional
        Number of realisations; taken from the data.

    Returns
    -------
    RichResult
        keys: ``ensemble_mean`` (length T), ``ensemble_std``,
        ``snr_gain`` (sqrt(M)), ``M``, ``T``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (ensemble averaging).
    """
    X = np.atleast_2d(np.asarray(x_k, dtype=float))
    m, T = X.shape
    if m < 1 or T < 1:
        raise ValueError("x_k must be a non-empty (M, T) array.")
    if M is not None and int(M) != m:
        raise ValueError(f"M = {M} does not match the {m} rows of x_k.")
    return RichResult(payload={"ensemble_mean": X.mean(axis=0),
                               "ensemble_std": X.std(axis=0),
                               "snr_gain": float(np.sqrt(m)), "M": int(m),
                               "T": int(T),
                               "method": "mean ACROSS realisations at each t; SNR gain sqrt(M)"})


def cheatsheet():
    return "rng018: ensemble mean is a function of t; SNR improves as sqrt(M)"
