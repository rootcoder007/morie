# morie.fn -- function file (rootcoder007/morie)
"""Sliding-blocks estimator of the extremal index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_extremal_sliding", "evt_extremal_index_slidblk"]


def ev_extremal_sliding(x, threshold=None, block_length=None):
    r"""Northrop's (2015) semiparametric maxima estimator of the
    extremal index, in its sliding-blocks form.

    The idea: for a stationary series with extremal index
    :math:`\theta`, the maximum of a block of :math:`b` observations
    satisfies :math:`P(M_b \le u) \approx F(u)^{b\theta}`. Estimate
    :math:`F` by the empirical distribution of the WHOLE series,
    transform each block maximum as
    :math:`Y_j = -b\log \hat F(M_j)`, and the :math:`Y_j` are
    approximately Exp(:math:`\theta`) -- so
    :math:`\hat\theta = 1/\bar Y`. SLIDING blocks (every window of
    length b, not just the disjoint ones) use each observation in b
    windows instead of one, and Northrop shows the sliding version
    has smaller asymptotic variance than the disjoint one -- the
    entire point of preferring it, and the tests compare the two.

    ``threshold`` is unused by this estimator and accepted only so
    the three extremal-index modules share a call shape; passing it
    changes nothing and the output says so.

    Parameters
    ----------
    x : array-like
        Stationary series.
    threshold : float, optional
        Ignored; present for signature parity with the runs and
        intervals estimators.
    block_length : int, optional
        Block size b; ``floor(sqrt(n))`` when omitted.

    Returns
    -------
    RichResult
        keys: ``theta``, ``theta_disjoint``, ``block_length``,
        ``n_sliding_blocks``, ``n_disjoint_blocks``,
        ``sliding_beats_disjoint_because``, ``n``, ``method``.

    References
    ----------
    Northrop, P. J. (2015), "An efficient semiparametric maxima
    estimator of the extremal index", *Extremes* 18:585-603,
    Secs. 2.2-2.3.
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 40:
        raise ValueError(f"need at least 40 observations, got {n}.")
    b = int(np.sqrt(n)) if block_length is None else int(block_length)
    if not 2 <= b <= n // 2:
        raise ValueError(f"block_length must lie in 2..{n // 2}, got {b}.")
    # empirical distribution of the whole series, right-continuous,
    # scaled by (n+1) so that log F_hat is finite at the sample max
    order = np.argsort(xv, kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    Fhat = ranks / (n + 1.0)

    def theta_from(maxF, bb):
        Y = -bb * np.log(maxF)
        m = float(Y.mean())
        if m <= 0:
            raise ValueError("degenerate block maxima; is the series "
                             "constant?")
        return min(1.0, 1.0 / m)

    # sliding: running max of Fhat over every window of length b
    from morie.fn._array_core import sliding_window_view

    slide_max = sliding_window_view(Fhat, b).max(axis=1)
    th_slide = theta_from(slide_max, b)
    nd = n // b
    disj_max = Fhat[:nd * b].reshape(nd, b).max(axis=1)
    th_disj = theta_from(disj_max, b)
    return RichResult(payload={
        "theta": float(th_slide), "theta_disjoint": float(th_disj),
        "block_length": b,
        "n_sliding_blocks": int(slide_max.size),
        "n_disjoint_blocks": int(nd),
        "sliding_beats_disjoint_because": (
            "every observation participates in b windows instead of one; "
            "Northrop (2015) shows the sliding estimator's asymptotic "
            "variance is strictly smaller"),
        "threshold_note": "this estimator uses block maxima, not a "
                          "threshold; the argument is accepted only for "
                          "signature parity and is ignored",
        "n": int(n),
        "method": "Northrop (2015) sliding-blocks semiparametric maxima "
                  "estimator of the extremal index"})


def cheatsheet():
    return "evextsl: -b log F_hat(block max) ~ Exp(theta); sliding blocks cut the variance"


#: Catalogue alias for :func:`ev_extremal_sliding`.
evt_extremal_index_slidblk = ev_extremal_sliding
