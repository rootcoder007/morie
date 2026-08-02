# morie.fn -- function file (rootcoder007/morie)
"""Autocorrelation estimate."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_acf_estimate"]


def rangayyan_acf_estimate(x, max_lag=None, biased=False):
    r"""Autocorrelation estimate (Rangayyan Ch. 3):

    .. math:: R_{xx}(m) = \frac{1}{N - |m|}
              \sum_{n=0}^{N-1-|m|} x(n)\,x(n+m).

    This is the UNBIASED estimator -- divisor N - |m|, not N. It is
    unbiased at every lag but its variance grows as |m| approaches N,
    and unlike the biased form it is not guaranteed positive
    semi-definite, so an AR fit from it can produce an unstable model.
    Both are returned and the trade-off is stated rather than hidden.

    Parameters
    ----------
    x : array-like
        Signal.
    max_lag : int, optional
        Maximum lag; N - 1 by default.
    biased : bool, default False
        Return the divisor-N form as the primary estimate.

    Returns
    -------
    RichResult
        keys: ``lags``, ``acf`` (per ``biased``), ``acf_unbiased``,
        ``acf_biased``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (autocorrelation).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N < 2:
        raise ValueError(f"need at least 2 samples, got {N}.")
    L = N - 1 if max_lag is None else int(max_lag)
    if not 0 <= L <= N - 1:
        raise ValueError(f"max_lag must lie in 0..{N - 1}, got {L}.")
    lags = np.arange(L + 1)
    raw = np.array([float(np.dot(x[: N - m], x[m:])) for m in lags])
    unb = raw / (N - lags)
    bia = raw / N
    return RichResult(payload={"lags": lags, "acf": bia if biased else unb,
                               "acf_unbiased": unb, "acf_biased": bia, "N": int(N),
                               "method": "R_xx(m) with divisor N-|m| (unbiased, not PSD-guaranteed)"})


def cheatsheet():
    return "rgacf: unbiased divisor N-|m|; biased form is PSD but shrinks toward 0"
