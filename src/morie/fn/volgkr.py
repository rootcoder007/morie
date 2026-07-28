# morie.fn -- function file (rootcoder007/morie)
"""Garman-Klass OHLC volatility estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_garman_klass"]


def vol_garman_klass(open_, high, low, close, periods_per_year=None):
    r"""The Garman-Klass (1980) "practical" estimator,

    .. math:: \hat\sigma^2 = \frac1n\sum_t\left[
              \frac12\left(\log\frac{H_t}{L_t}\right)^2
              - (2\log 2 - 1)\left(\log\frac{C_t}{O_t}\right)^2
              \right],

    their :math:`\hat\sigma^2_{*}` (Eq. (20)): the minimum-variance
    unbiased combination of the squared log-range and the squared
    open-to-close return for driftless Brownian motion, with
    efficiency about 7.4 relative to close-to-close -- better than
    Parkinson's 4.9 because the open and close carry information the
    range alone does not.

    The subtraction is the part that surprises: the open-close term
    enters NEGATIVELY (:math:`2\log 2 - 1 \approx 0.386`), because
    given the range, a large open-to-close move indicates trend
    rather than volatility, and the optimal combination partials it
    out. A single bar can therefore produce a negative variance
    estimate; the average over bars is what the theory speaks about,
    and if THAT is negative the data are telling you the model
    (driftless diffusion, no jumps, no gaps) does not hold -- an
    error here, not a clipped zero. The same drift and discreteness
    caveats as Parkinson apply, plus sensitivity to opening gaps
    since :math:`O_t` is used as the bar's origin.

    Parameters
    ----------
    open_, high, low, close : array-like
        Per-bar OHLC, positive, ``low <= open_, close <= high``.
    periods_per_year : float, optional
        When given, an annualised sigma is also returned.

    Returns
    -------
    RichResult
        keys: ``variance``, ``sigma``, ``sigma_annualised``,
        ``range_term``, ``openclose_term``, ``efficiency_vs_close``,
        ``negative_bar_fraction``, ``n``, ``method``.

    References
    ----------
    Garman, M. B. and Klass, M. J. (1980), "On the estimation of
    security price volatilities from historical data", *Journal of
    Business* 53:67-78, Eq. (20).
    """
    O = np.asarray(open_, dtype=float).ravel()
    H = np.asarray(high, dtype=float).ravel()
    L = np.asarray(low, dtype=float).ravel()
    C = np.asarray(close, dtype=float).ravel()
    n = O.size
    if not (H.size == L.size == C.size == n):
        raise ValueError("open, high, low and close must share a length.")
    if n < 2:
        raise ValueError(f"need at least 2 bars, got {n}.")
    if np.any(L <= 0):
        raise ValueError("prices must be positive.")
    if np.any((H < L) | (O > H) | (O < L) | (C > H) | (C < L)):
        raise ValueError("each bar needs low <= open, close <= high.")
    hl = np.log(H / L) ** 2
    co = np.log(C / O) ** 2
    per_bar = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    var = float(per_bar.mean())
    if var <= 0:
        raise ValueError(
            "the average Garman-Klass variance is not positive: the "
            "driftless-diffusion model this estimator assumes does not "
            "describe these bars (strong trend, jumps or gaps).")
    sig = float(np.sqrt(var))
    return RichResult(payload={
        "variance": var, "sigma": sig,
        "sigma_annualised": (sig * np.sqrt(float(periods_per_year))
                             if periods_per_year else None),
        "range_term": float(np.mean(0.5 * hl)),
        "openclose_term": float(np.mean((2 * np.log(2) - 1) * co)),
        "negative_sign_note": "the open-close term enters NEGATIVELY: given "
                              "the range, a large open-to-close move signals "
                              "trend, not volatility, and the optimal "
                              "combination partials it out",
        "efficiency_vs_close": 7.4,
        "negative_bar_fraction": float(np.mean(per_bar < 0)),
        "gap_caveat": "O_t is the bar's origin, so overnight gaps leak into "
                      "nothing here -- and are therefore missed entirely",
        "n": int(n),
        "method": "Garman-Klass (1980) Eq. (20): 0.5 (log H/L)^2 - (2 log2 - 1)(log C/O)^2"})


def cheatsheet():
    return "volgkr: the open-close term is SUBTRACTED -- range minus trend, efficiency 7.4"
