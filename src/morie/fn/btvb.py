# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap variance from replicates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boot_var_estimator"]


def boot_var_estimator(theta_b):
    r"""The bootstrap variance of a statistic from its replicates,

    .. math:: \widehat{\mathrm{Var}} = \frac1{B-1}\sum_b
              (\hat\theta^*_b - \bar\theta^*)^2

    (Efron and Tibshirani 1993, Eq. (6.5); the same B - 1 as ESL's
    (7.53), because the replicates are centred at their own mean).
    This module takes the REPLICATES, not the data: it is the second
    half of a bootstrap someone has already run, split out so that
    expensive statistics need resampling only once for variance,
    bias (``morie.fn.btbias``) and intervals together.

    Parameters
    ----------
    theta_b : array-like
        Bootstrap replicates of the statistic.

    Returns
    -------
    RichResult
        keys: ``value``, ``se``, ``mean_replicate``, ``B``,
        ``denominator``, ``method``.

    References
    ----------
    Efron, B. and Tibshirani, R. J. (1993), *An Introduction to the
    Bootstrap*, Chapman and Hall, Ch. 6, Eq. (6.5). Efron (1979).
    """
    r = np.asarray(theta_b, dtype=float).ravel()
    B = r.size
    if B < 2:
        raise ValueError(f"need at least 2 replicates, got {B}.")
    if not np.all(np.isfinite(r)):
        raise ValueError("every replicate must be finite; a failed "
                         "refit should be dropped before this point, "
                         "and dropping it changes the estimand.")
    v = float(np.var(r, ddof=1))
    return RichResult(payload={
        "value": v, "se": float(np.sqrt(v)),
        "mean_replicate": float(r.mean()),
        "B": int(B), "denominator": "B - 1",
        "denominator_note": "B - 1, not B: the replicates are centred at "
                            "their own mean (Efron-Tibshirani Eq. 6.5)",
        "method": "Bootstrap variance from replicates, Efron-Tibshirani (6.5)"})


def cheatsheet():
    return "btvb: variance of the replicates over B - 1 -- takes replicates, not data"
