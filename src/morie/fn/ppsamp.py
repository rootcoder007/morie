# morie.fn -- function file (rootcoder007/morie)
"""Probability-proportional-to-size inclusion probabilities and estimators."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pps_sampling"]


def pps_sampling(y, size, n):
    """PPS inclusion probabilities with the two estimators they support.

    Two different estimators are reported because they answer to
    different designs from the same size measure.  Hansen-Hurwitz is the
    with-replacement estimator built on the selection probabilities
    ``p_i = x_i / X``; Horvitz-Thompson is the without-replacement
    estimator built on the inclusion probabilities ``pi_i = n p_i``.  A
    unit whose size would give ``pi_i > 1`` cannot be sampled without
    replacement at that rate at all -- it must be taken with certainty --
    so that case is an error here rather than a silently truncated
    probability.

    Formula: ``p_i = x_i / sum_k x_k``; ``pi_i = n p_i``;
    ``Yhat_HH = (1/n) sum_i y_i / p_i``;
    ``Yhat_HT = sum_i y_i / pi_i``.

    Parameters
    ----------
    y : array-like
        Values for the sampled units.
    size : array-like
        Positive size measure, same length as ``y``.
    n : int
        Sample size, at least 1.

    Returns
    -------
    RichResult
        ``pi`` (inclusion probabilities), ``p`` (selection
        probabilities), ``estimate`` (the Hansen-Hurwitz total),
        ``ht_total``, ``se`` (the Hansen-Hurwitz standard error),
        ``X`` (total size), ``n``.

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
        raise ValueError("pps_sampling: size must have one entry per observation")
    for v in x:
        if v <= 0.0:
            raise ValueError("pps_sampling: sizes must be positive")
    n = int(n)
    if n < 1:
        raise ValueError("pps_sampling: n must be at least 1")
    X = 0.0
    for v in x:
        X += v
    p = [v / X for v in x]
    pi = [n * v for v in p]
    for v in pi:
        if v > 1.0:
            raise ValueError("pps_sampling: an inclusion probability exceeds 1; "
                             "that unit must be selected with certainty")
    zs = [y[i] / p[i] for i in range(len(y))]
    hh = 0.0
    for v in zs:
        hh += v
    hh /= n
    ht = 0.0
    for i in range(len(y)):
        ht += y[i] / pi[i]
    se = float("nan")
    if len(y) > 1:
        ss = 0.0
        for v in zs:
            ss += (v - hh) * (v - hh)
        se = (ss / ((len(y) - 1) * n)) ** 0.5
    return RichResult(payload={
        "pi": pi, "p": p, "estimate": hh, "hh_total": hh, "ht_total": ht,
        "se": se, "X": X, "n": n, "method": "PPS: Hansen-Hurwitz and Horvitz-Thompson"})


def cheatsheet():
    return "ppsamp: Probability proportional to size sampling"
