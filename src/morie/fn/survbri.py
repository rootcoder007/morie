# morie.fn -- function file (rootcoder007/morie)
"""Brier score for survival prediction."""

from .brier import brier

from ._richresult import RichResult

__all__ = ["brier_score"]


def brier_score(time, event, predicted_S, t_grid):
    """IPCW Brier score, ``E[(I(T > t) - S(t|X))^2]``.

    Censoring makes the naive average biased, because subjects censored
    before the horizon have unknown status.  Graf et al. reweight the
    observed subjects by the inverse censoring survival at their own
    event time, which restores unbiasedness.  That estimator already
    exists in the tree, so this module is a thin alias for
    ``brier.brier`` rather than a second implementation.

    Formula: ``BS(t) = E[(I(T > t) - S(t|X))^2]``, IPCW-weighted.

    Parameters
    ----------
    time : array-like
        Observed event or censoring times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.
    predicted_S : array-like
        Predicted survival at ``t_grid``: length ``n``, or ``n x k``.
    t_grid : float or array-like
        Evaluation time(s).

    Returns
    -------
    RichResult
        ``estimate`` (the Brier score), ``brier_score``, ``scaled_brier``,
        ``integrated_brier``, ``eval_time``, ``method``.

    References
    ----------
    Graf, E., Schmoor, C., Sauerbrei, W. & Schumacher, M. (1999).
    Assessment and comparison of prognostic classification schemes for
    survival data.  Statistics in Medicine 18(17-18):2529-2545.
    Gerds, T. A. & Schumacher, M. (2006).  Biometrical Journal
    48(6):1029-1040.  <https://doi.org/10.1002/bimj.200610301>
    """
    r = brier(time, event, predicted_S, t_grid)
    return RichResult(payload={
        "estimate": r["brier_score"], "brier_score": r["brier_score"],
        "scaled_brier": r["scaled_brier"],
        "integrated_brier": r["integrated_brier"],
        "eval_time": r["eval_time"],
        "method": "IPCW Brier score [Graf et al. 1999; Gerds & Schumacher 2006]"})


# CANONICAL TEST
# >>> # everyone survives past the horizon and the model says so: BS is exactly 0
# >>> r = brier_score([5.0, 6.0, 7.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 2.0)
# >>> assert abs(r["estimate"]) < 1e-15


def cheatsheet():
    return "survbri(time, event, predicted_S, t_grid): IPCW Brier (alias of brier)."
