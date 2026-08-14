# morie.fn -- function file (rootcoder007/morie)
"""Weight trimming at an absolute threshold, with the excess redistributed."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["weight_trimming"]


def weight_trimming(y, weights, threshold):
    """Cap the weights at ``threshold`` and give the excess back to the rest.

    Plain truncation loses weighted mass and therefore biases any
    population total downwards.  Potter's redistribution step puts the
    removed mass back onto the units that were not trimmed, in
    proportion to their own weights, so the weighted total is preserved
    exactly.  Redistribution can push a previously untrimmed unit above
    the threshold, so the cap-and-redistribute pair is iterated until no
    weight exceeds it.

    Formula: ``w_i' = min(w_i, w_max)``, then the removed mass
    ``sum_i (w_i - w_i')`` is spread over the untrimmed units in
    proportion to ``w_i'``; repeat.

    Parameters
    ----------
    y : array-like
        Observations, used only to report the effect of trimming on the
        weighted mean.
    weights : array-like
        Design weights, non-negative, same length as ``y``.
    threshold : float
        Maximum permitted weight; must be at least the mean weight, else
        no feasible redistribution exists.

    Returns
    -------
    RichResult
        ``estimate`` (weighted mean after trimming), ``mean_before``,
        ``weights`` (trimmed), ``n_trimmed``, ``iterations``, ``sumw``,
        ``max_weight``, ``deff_before``, ``deff_after``, ``n``, ``method``.

    References
    ----------
    Potter, F. J. (1990).  A study of procedures to identify and trim
    extreme sampling weights.  Proceedings of the Section on Survey
    Research Methods, American Statistical Association, 225-230.
    """
    yy = C.vec(y)
    w = C.vec(weights)
    n = len(yy)
    if n == 0:
        raise ValueError("weight_trimming: y is empty")
    if len(w) != n:
        raise ValueError("weight_trimming: y and weights differ in length")
    for v in w:
        if v < 0.0:
            raise ValueError("weight_trimming: weights must be non-negative")
    thr = float(threshold)
    tot = sum(w)
    if thr <= 0.0:
        raise ValueError("weight_trimming: threshold must be positive")
    if thr * n < tot:
        raise ValueError("weight_trimming: threshold below the mean weight, no feasible redistribution")
    mu0 = sum(w[i] * yy[i] for i in range(n)) / tot
    cur = list(w)
    it = 0
    for it in range(1, 101):
        over = sum(v - thr for v in cur if v > thr)
        if over <= 1e-15:
            it -= 1
            break
        cur = [thr if v > thr else v for v in cur]
        free = sum(v for v in cur if v < thr)
        if free <= 0.0:
            break
        cur = [v if v >= thr else v + over * v / free for v in cur]
    ntr = sum(1 for i in range(n) if w[i] > thr)
    s1 = sum(cur)
    mu1 = sum(cur[i] * yy[i] for i in range(n)) / s1
    return RichResult(payload={
        "estimate": float(mu1), "mean_before": float(mu0), "weights": cur,
        "n_trimmed": int(ntr), "iterations": int(it), "sumw": float(s1),
        "max_weight": float(max(cur)),
        "deff_before": _deff(w), "deff_after": _deff(cur), "n": n,
        "method": "cap at w_max then redistribute the excess [Potter 1990]"})


def _deff(w):
    n = len(w)
    s = sum(w)
    if s <= 0.0:
        return float("nan")
    return n * sum(v * v for v in w) / (s * s)


# CANONICAL TEST
# >>> r = weight_trimming([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 9.0], 4.0)
# >>> assert abs(r["sumw"] - 12.0) < 1e-12          # mass is preserved
# >>> assert r["max_weight"] <= 4.0 + 1e-12
# >>> assert r["deff_after"] <= r["deff_before"]
# >>> # a threshold above every weight is the identity
# >>> u = weight_trimming([1.0, 2.0], [1.0, 2.0], 10.0)
# >>> assert u["n_trimmed"] == 0 and abs(u["estimate"] - u["mean_before"]) < 1e-12


def cheatsheet():
    return "trimit(y, weights, threshold): cap weights, redistribute the excess."

# public names resolved by fn/_lazy_map.json
weighttrimming = weight_trimming
