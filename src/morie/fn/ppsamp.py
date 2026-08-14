# morie.fn -- function file (rootcoder007/morie)
"""Probability-proportional-to-size selection probabilities and design variance."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pps_sampling"]


def pps_sampling(y, size, n):
    """PPS design quantities for a frame of units.

    ``y`` and ``size`` describe the FRAME -- every unit that could be
    drawn -- and ``n`` is the number of draws.  That is the only reading
    under which ``pi_i = n x_i / sum_k x_k`` is a probability at all: the
    denominator is the population size total, not a sample total.  A unit
    with ``pi_i > 1`` cannot be included with that probability without
    replacement and must be taken with certainty, so it is an error here
    rather than a silently truncated probability.

    The design variance of the with-replacement (Hansen-Hurwitz)
    estimator is available in closed form from the frame, and it is
    exactly zero when ``y`` is proportional to ``size`` -- the case PPS
    is designed for, and the anchor used for this module.

    Formula: ``p_i = x_i / X``; ``pi_i = n p_i``;
    ``Var(Yhat_HH) = (1/n) sum_i p_i (y_i / p_i - Y)^2``.

    Parameters
    ----------
    y : array-like
        Values for the frame units.
    size : array-like
        Positive size measure, same length as ``y``.
    n : int
        Number of draws, at least 1.

    Returns
    -------
    RichResult
        ``pi`` (inclusion probabilities), ``p`` (selection
        probabilities), ``estimate`` (the frame total ``Y``),
        ``hh_variance``, ``hh_se``, ``srs_variance`` (the with-
        replacement variance of the equal-probability design on the same
        frame), ``deff`` (their ratio), ``X``, ``N``, ``n``.

    References
    ----------
    Hansen, M. H. & Hurwitz, W. N. (1943).  On the theory of sampling
    from finite populations.  Annals of Mathematical Statistics
    14(4):333-362.  doi:10.1214/aoms/1177731356.
    """
    y = [float(v) for v in C.vec(y)]
    x = [float(v) for v in C.vec(size)]
    if len(y) == 0:
        raise ValueError("pps_sampling: y is empty")
    if len(x) != len(y):
        raise ValueError("pps_sampling: size must have one entry per unit")
    for v in x:
        if v <= 0.0:
            raise ValueError("pps_sampling: sizes must be positive")
    n = int(n)
    if n < 1:
        raise ValueError("pps_sampling: n must be at least 1")
    N = len(y)
    X = 0.0
    for v in x:
        X += v
    p = [v / X for v in x]
    pi = [n * v for v in p]
    for v in pi:
        if v > 1.0:
            raise ValueError("pps_sampling: an inclusion probability exceeds 1; "
                             "that unit must be selected with certainty")
    Y = 0.0
    for v in y:
        Y += v
    var = 0.0
    for i in range(N):
        t = y[i] / p[i] - Y
        var += p[i] * t * t
    var /= n
    svar = 0.0
    for i in range(N):
        t = N * y[i] - Y
        svar += t * t / N
    svar /= n
    return RichResult(payload={
        "pi": pi, "p": p, "estimate": Y, "total": Y,
        "hh_variance": var, "hh_se": var ** 0.5,
        "srs_variance": svar,
        "deff": var / svar if svar > 0.0 else float("nan"),
        "X": X, "N": N, "n": n,
        "method": "PPS selection probabilities and Hansen-Hurwitz design variance"})


def cheatsheet():
    return "ppsamp: Probability proportional to size sampling"

# public names resolved by fn/_lazy_map.json
ppssampling = pps_sampling
