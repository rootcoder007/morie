# morie.fn -- function file (rootcoder007/morie)
"""Madogram estimator of the Pickands dependence function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ev_madogram"]


def ev_madogram(x, y, t=None):
    r"""The lambda-madogram estimate of the Pickands dependence
    function :math:`A(t)` of a bivariate extreme-value copula
    (Naveau, Guillou, Cooley and Diebolt 2009).

    With :math:`F, G` the margins (estimated by ranks), the
    lambda-madogram is

    .. math:: \nu(t) = \frac12 E\left|F(X)^{1/t}
              - G(Y)^{1/(1-t)}\right|,

    and their Prop. 3 inverts it:

    .. math:: \hat A(t) = \frac{\nu(t) + c(t)}{1 - \nu(t) - c(t)},
              \qquad c(t) = \frac{t}{2(1+t)} +
              \frac{1-t}{2(2-t)} .

    :math:`A` is what dependence between extremes looks like:
    :math:`A \equiv 1` is asymptotic independence,
    :math:`A(t) = \max(t, 1-t)` complete dependence, and every valid
    :math:`A` is convex between those envelopes with
    :math:`A(0) = A(1) = 1`. The estimate is clipped into the
    envelope, as the authors recommend, and the clipping is reported
    rather than silent -- an estimate that needed heavy clipping is
    telling you the extreme-value model itself fits badly.

    Parameters
    ----------
    x, y : array-like
        Paired observations.
    t : array-like, optional
        Evaluation points in (0, 1); a grid when omitted.

    Returns
    -------
    RichResult
        keys: ``t``, ``A``, ``A_raw``, ``clipped_fraction``,
        ``dependence_summary`` (2(1 - A(1/2)), in [0, 1]),
        ``envelope``, ``n``, ``method``.

    References
    ----------
    Naveau, P., Guillou, A., Cooley, D. and Diebolt, J. (2009),
    "Modelling pairwise dependence of maxima in space",
    *Biometrika* 96:1-17, Prop. 3. Pickands (1981) for A itself.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if xv.size != yv.size:
        raise ValueError(f"x has {xv.size} entries and y has {yv.size}.")
    n = xv.size
    if n < 20:
        raise ValueError(f"need at least 20 pairs, got {n}.")
    tg = np.linspace(0.05, 0.95, 19) if t is None else \
        np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    if np.any((tg <= 0) | (tg >= 1)):
        raise ValueError("t must lie strictly in (0, 1).")
    # rank margins, scaled by n+1 to stay inside (0, 1)
    U = (np.argsort(np.argsort(xv)) + 1.0) / (n + 1.0)
    V = (np.argsort(np.argsort(yv)) + 1.0) / (n + 1.0)
    A_raw = np.empty(tg.size)
    for i, tt in enumerate(tg):
        nu = 0.5 * float(np.mean(np.abs(U ** (1.0 / tt)
                                        - V ** (1.0 / (1.0 - tt)))))
        c = tt / (2.0 * (1.0 + tt)) + (1.0 - tt) / (2.0 * (2.0 - tt))
        A_raw[i] = (nu + c) / (1.0 - nu - c)
    lower = np.maximum(tg, 1.0 - tg)
    A = np.clip(A_raw, lower, 1.0)
    clipped = float(np.mean(np.abs(A - A_raw) > 1e-12))
    # 2(1 - A(1/2)) in [0, 1]: 0 = independence, 1 = complete dependence
    nu_h = 0.5 * float(np.mean(np.abs(U ** 2 - V ** 2)))
    c_h = 0.5 / 3.0 + 0.5 / 3.0
    A_half = np.clip((nu_h + c_h) / (1.0 - nu_h - c_h), 0.5, 1.0)
    return RichResult(payload={
        "t": tg, "A": A, "A_raw": A_raw,
        "clipped_fraction": clipped,
        "dependence_summary": float(2.0 * (1.0 - A_half)),
        "envelope": "max(t, 1-t) <= A <= 1; A = 1 is asymptotic "
                    "independence, the lower envelope complete dependence",
        "clipping_note": "heavy clipping means the extreme-value model "
                         "itself fits badly, not that the estimator "
                         "misfired",
        "n": int(n),
        "method": "Lambda-madogram estimate of the Pickands dependence "
                  "function (Naveau et al. 2009, Prop. 3)"})


def cheatsheet():
    return "evmadog: A from the lambda-madogram -- clip to the envelope, and report the clipping"
