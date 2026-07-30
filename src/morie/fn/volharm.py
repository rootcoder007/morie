# morie.fn -- function file (rootcoder007/morie)
"""Harmonic-mean aggregation of per-period volatilities."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_harmonic", "vol_harmonic_volatility"]


def vol_harmonic(sigma):
    r"""Harmonic-mean aggregate of a series of per-period volatility
    estimates,

    .. math:: \bar\sigma_H = \frac{n}{\sum_t 1/\sigma_t},

    reported alongside the geometric and arithmetic means with the
    inequality :math:`\bar\sigma_H \le \bar\sigma_G \le \bar\sigma_A`
    (AM-GM-HM) asserted rather than assumed.

    When each aggregate is the right one is the substance. The
    ARITHMETIC mean of VARIANCES is what realized-volatility theory
    aggregates -- integrated variance is a sum (Andersen,
    Bollerslev, Diebold and Labys 2003) -- so for combining
    sub-period variances into a total, use arithmetic on
    :math:`\sigma^2`, and that value is returned too. The HARMONIC
    mean is the right average when the quantity enters downstream
    through its RECIPROCAL: precision weighting (weights
    :math:`1/\sigma_t^2`), time-to-target calculations, and averaging
    of rates. Because the harmonic mean is dominated by the SMALLEST
    values, it is also the conservative choice when occasional
    spuriously large :math:`\sigma_t` (data glitches) are the
    contamination of concern -- and the worst choice when spuriously
    small ones are. That asymmetry is stated, not hidden.

    Parameters
    ----------
    sigma : array-like
        Positive per-period volatilities.

    Returns
    -------
    RichResult
        keys: ``harmonic``, ``geometric``, ``arithmetic``,
        ``rms`` (the sqrt of the arithmetic mean variance),
        ``inequality_holds``, ``which_to_use``, ``n``, ``method``.

    References
    ----------
    Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P.
    (2003), "Modeling and forecasting realized volatility",
    *Econometrica* 71:579-625, for variance aggregation. Standard
    AM-GM-HM inequality for the ordering.
    """
    s = np.asarray(sigma, dtype=float).ravel()
    n = s.size
    if n < 1:
        raise ValueError("need at least one volatility.")
    if np.any(s <= 0):
        raise ValueError("volatilities must be positive; a zero makes the "
                         "harmonic mean zero regardless of everything else.")
    hm = float(n / np.sum(1.0 / s))
    gm = float(np.exp(np.mean(np.log(s))))
    am = float(np.mean(s))
    rms = float(np.sqrt(np.mean(s ** 2)))
    return RichResult(payload={
        "harmonic": hm, "geometric": gm, "arithmetic": am, "rms": rms,
        "inequality_holds": bool(hm <= gm + 1e-12 and gm <= am + 1e-12),
        "which_to_use": "arithmetic on VARIANCES (the rms here) for "
                        "aggregating sub-period volatility into a total -- "
                        "integrated variance is a sum (ABDL 2003); harmonic "
                        "when the quantity enters through its reciprocal "
                        "(precision weights, rates)",
        "contamination_asymmetry": "the harmonic mean is dominated by the "
                                   "SMALLEST values: robust to spuriously "
                                   "large sigmas, worst-case for spuriously "
                                   "small ones",
        "n": int(n),
        "method": "Harmonic / geometric / arithmetic / rms volatility aggregates"})


def cheatsheet():
    return "volharm: HM <= GM <= AM always -- and integrated variance aggregates ARITHMETICALLY"


#: Catalogue alias for :func:`vol_harmonic`.
vol_harmonic_volatility = vol_harmonic
