# morie.fn -- function file (rootcoder007/morie)
"""Sample life table estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sample_lifetable"]


def sample_lifetable(intervals, entered, died, withdrawn=None):
    r"""Actuarial (grouped) life-table estimator:

    .. math:: \hat q_j = \frac{d_j}{n_j - w_j/2},
              \qquad
              \hat S_j = \prod_{k \le j}(1 - \hat q_k).

    Built for INTERVAL-grouped data, where exact times are unknown --
    registry tables, published cohort summaries. The
    :math:`w_j/2` is the actuarial correction: withdrawals are
    assumed uniform within the interval, so on average each
    contributes half an interval of exposure.

    That assumption is the method's only real content, and it is
    checkable in spirit rather than in the data: with wide intervals
    and heavy withdrawal it fails, and the estimate is biased in a
    direction that depends on whether withdrawal concentrates early
    or late. Kaplan-Meier is the limit as intervals shrink, which is
    why it should be preferred whenever exact times exist.

    Parameters
    ----------
    intervals : array-like
        Interval boundaries, increasing, length J+1.
    entered : array-like
        Number entering each interval.
    died : array-like
        Deaths in each interval.
    withdrawn : array-like, optional
        Withdrawals in each interval.

    Returns
    -------
    RichResult
        keys: ``intervals``, ``q``, ``survival``, ``effective_n``,
        ``actuarial_correction``, ``assumes``, ``J``, ``method``.
    """
    edges = np.atleast_1d(np.asarray(intervals, dtype=float)).ravel()
    if edges.size < 2:
        raise ValueError("need at least 2 interval boundaries.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("interval boundaries must be strictly increasing.")
    J = edges.size - 1
    nj = np.atleast_1d(np.asarray(entered, dtype=float)).ravel()
    dj = np.atleast_1d(np.asarray(died, dtype=float)).ravel()
    wj = np.zeros(J) if withdrawn is None else \
        np.atleast_1d(np.asarray(withdrawn, dtype=float)).ravel()
    for nm, arr in (("entered", nj), ("died", dj), ("withdrawn", wj)):
        if arr.size != J:
            raise ValueError(f"{nm} has {arr.size} entries for {J} intervals.")
        if np.any(arr < 0):
            raise ValueError(f"{nm} must be non-negative.")
    eff = nj - wj / 2.0
    if np.any(eff <= 0):
        raise ValueError("effective sample size is non-positive in some "
                         "interval; check entered against withdrawn.")
    if np.any(dj > eff):
        raise ValueError("more deaths than effective exposure in some interval.")
    q = dj / eff
    S = np.cumprod(1.0 - q)
    return RichResult(payload={
        "intervals": edges, "q": q, "survival": S, "effective_n": eff,
        "actuarial_correction": "n_j - w_j/2",
        "assumes": "withdrawals uniform within each interval, so each "
                   "contributes half an interval of exposure on average",
        "prefer": "Kaplan-Meier whenever exact times exist; this is its "
                  "grouped-data limit",
        "J": int(J),
        "method": "Actuarial life table for interval-grouped data"})


def cheatsheet():
    return "smplts: the w/2 IS the method -- uniform withdrawal within the interval, nothing more"
