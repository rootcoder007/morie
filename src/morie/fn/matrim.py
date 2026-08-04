# morie.fn -- function file (rootcoder007/morie)
"""Duval-Tweedie trim and fill."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ma_trim_fill"]


def ma_trim_fill(yi, vi, side="left", n_iter=50):
    """Adjust a pooled effect for suppressed small studies.

    A funnel plot that is asymmetric suggests studies are missing from
    one side.  Trim and fill takes that literally: it strips the
    asymmetric tail, re-estimates the centre from what is left, then
    puts back mirror images of the trimmed studies so the funnel is
    symmetric again.  The filled studies are fabrications and the method
    is a sensitivity analysis, not an estimate of what those studies
    would have found -- the point is how far the pooled effect moves,
    not where it lands.

    Determinism: a fixed number of trim-refit rounds, with ties in the
    ranks broken by position, so nothing depends on a stopping rule.

    Formula: with ``T_n`` the sum of ranks of ``|y_i - theta|`` over the
    studies on the suppressed-opposite side, the ``L0`` estimator of the
    number missing is ``L0 = (4 T_n - n (n + 1)) / (2 n - 1)``; filled
    values are ``2 theta - y_i``.

    Parameters
    ----------
    yi : array-like, shape (k,)
        Study effect sizes.
    vi : array-like, shape (k,)
        Study sampling variances.
    side : str, default "left"
        Side the missing studies are on: ``"left"`` or ``"right"``.
    n_iter : int, default 50
        Trim-and-refit rounds.

    Returns
    -------
    RichResult
        ``theta_adj`` (pooled effect after filling), ``k_filled``,
        ``fill_yi``, ``theta_raw``, ``estimate`` (same as
        ``theta_adj``), ``k``.

    References
    ----------
    Duval, S. & Tweedie, R. (2000).  Trim and fill: a simple
    funnel-plot-based method of testing and adjusting for publication
    bias in meta-analysis.  Biometrics 56:455-463.  The ``L0``
    estimator is their equation (2); the iterative trim-and-refit
    scheme is section 3 of the same paper.
    """
    y = C.vec(yi)
    v = C.vec(vi)
    k = len(y)
    sgn = 1.0 if side == "left" else -1.0
    w = [1.0 / t for t in v]
    theta = sum(w[i] * y[i] for i in range(k)) / sum(w)
    theta_raw = theta
    k0 = 0
    for _ in range(int(n_iter)):
        c = [sgn * (y[i] - theta) for i in range(k)]
        r = S.rank_first([abs(t) for t in c])
        Tn = sum(r[i] for i in range(k) if c[i] > 0.0)
        l0 = (4.0 * Tn - k * (k + 1.0)) / (2.0 * k - 1.0)
        k0 = int(S.rnd(l0))
        if k0 < 0:
            k0 = 0
        if k0 > k - 1:
            k0 = k - 1
        o = S.order([sgn * y[i] for i in range(k)])
        keep = o[:k - k0]
        sw = sum(w[i] for i in keep)
        theta = sum(w[i] * y[i] for i in keep) / sw
    o = S.order([sgn * y[i] for i in range(k)])
    trimmed = o[k - k0:]
    fill = [2.0 * theta - y[i] for i in trimmed]
    fill_v = [v[i] for i in trimmed]
    ally = y + fill
    allv = v + fill_v
    allw = [1.0 / t for t in allv]
    theta_adj = sum(allw[i] * ally[i] for i in range(len(ally))) / sum(allw)
    return RichResult(payload={
        "theta_adj": theta_adj, "estimate": theta_adj, "k_filled": k0,
        "fill_yi": fill, "theta_raw": theta_raw, "k": k,
        "method": "Duval-Tweedie trim and fill, L0 estimator"})


matrimfill = ma_trim_fill


def cheatsheet():
    return "matrim: Duval-Tweedie trim and fill."
