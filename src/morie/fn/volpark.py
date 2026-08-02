# morie.fn -- function file (rootcoder007/morie)
"""Parkinson high-low range volatility estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_parkinson", "vol_parkinson_range"]


def vol_parkinson(high, low, periods_per_year=None):
    r"""Parkinson's (1980) extreme-value volatility estimator,

    .. math:: \hat\sigma^2 = \frac1{4\log 2}\cdot\frac1n\sum_t
              \left(\log\frac{H_t}{L_t}\right)^2 .

    The constant :math:`1/(4\log 2)` is not a fudge factor: for
    driftless Brownian motion the expected squared log-range is
    :math:`4\log 2\,\sigma^2` (Parkinson's Eq. (4), from Feller's
    range distribution), and dividing by it makes the estimator
    unbiased. Using the range instead of close-to-close returns buys
    about a 4.9-fold variance reduction -- the range sees the whole
    path where the close sees one point -- and the tests measure that
    ratio rather than quoting it.

    Two known biases, both stated: DRIFT inflates the range, so the
    estimator over-reads trending periods (the derivation assumes
    zero drift); and DISCRETE sampling of the true path means the
    observed high and low understate the true extremes, biasing the
    estimate down for coarsely sampled bars.

    Parameters
    ----------
    high, low : array-like
        Per-bar highs and lows, positive, ``high >= low``.
    periods_per_year : float, optional
        When given, an annualised sigma is also returned.

    Returns
    -------
    RichResult
        keys: ``variance``, ``sigma``, ``sigma_annualised``,
        ``constant``, ``efficiency_vs_close``, ``drift_bias``,
        ``discreteness_bias``, ``n``, ``method``.

    References
    ----------
    Parkinson, M. (1980), "The extreme value method for estimating
    the variance of the rate of return", *Journal of Business*
    53:61-65, Eq. (4).
    """
    H = np.asarray(high, dtype=float).ravel()
    L = np.asarray(low, dtype=float).ravel()
    if H.size != L.size:
        raise ValueError(f"high has {H.size} entries and low has {L.size}.")
    n = H.size
    if n < 2:
        raise ValueError(f"need at least 2 bars, got {n}.")
    if np.any(L <= 0):
        raise ValueError("prices must be positive.")
    if np.any(H < L):
        raise ValueError("high must be at least low in every bar.")
    const = 1.0 / (4.0 * np.log(2.0))
    var = const * float(np.mean(np.log(H / L) ** 2))
    sig = float(np.sqrt(var))
    return RichResult(payload={
        "variance": var, "sigma": sig,
        "sigma_annualised": (sig * np.sqrt(float(periods_per_year))
                             if periods_per_year else None),
        "constant": const,
        "constant_note": "1/(4 log 2): E[(log range)^2] = 4 log2 sigma^2 "
                         "for driftless Brownian motion (Parkinson Eq. 4)",
        "efficiency_vs_close": 4.9,
        "drift_bias": "drift inflates the range, so trending periods read "
                      "high; the derivation assumes zero drift",
        "discreteness_bias": "the observed high/low of a discretely sampled "
                             "path understate the true extremes, biasing "
                             "the estimate down for coarse bars",
        "n": int(n),
        "method": "Parkinson (1980) range estimator, 1/(4 log 2) mean squared log-range"})


def cheatsheet():
    return "volpark: (log H/L)^2 / (4 log 2) -- unbiased for driftless BM, and drift reads high"


#: Catalogue alias for :func:`vol_parkinson`.
vol_parkinson_range = vol_parkinson
