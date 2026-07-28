# morie.fn -- function file (rootcoder007/morie)
"""Sample RMS."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sample_rms"]


def rangayyan_ch3_sample_rms(eta, N=None):
    r"""Root-mean-square value (Rangayyan Ch. 3):

    .. math:: RMS_\eta = \sqrt{\frac1N \sum_{n=0}^{N-1}
              [\eta(n)]^2}.

    Parameters
    ----------
    eta : array-like
        Samples.
    N : int, optional
        Length.

    Returns
    -------
    RichResult
        keys: ``rms``, ``mean_square``, ``N``, ``method``.
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
    ms = float(np.mean(eta**2))
    return RichResult(payload={"rms": float(np.sqrt(ms)), "mean_square": ms,
                               "N": int(eta.size),
                               "method": "RMS = sqrt((1/N) sum eta^2)"})


def cheatsheet():
    return "rng009: RMS = sqrt(mean square)"
