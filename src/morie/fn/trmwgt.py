# morie.fn -- function file (rootcoder007/morie)
"""Weight truncation at an upper percentile of the weight distribution."""

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["trim_weights"]


def trim_weights(weights, quantile=0.99):
    """Truncate design weights at a percentile of their own distribution.

    Potter's simplest procedure: pick a cut point from the empirical
    weight distribution and pull every weight above it down onto the cut.
    Truncation lowers the variance of the weighted estimator and biases
    it, so the mass removed is reported explicitly and a rescaled copy
    that restores the original total is returned alongside.

    The cut point is the type-7 quantile (R's ``quantile()`` default) so
    both language arms land on the same number.

    Formula: ``w_i' = min(w_i, Q_q(w))``.

    Parameters
    ----------
    weights : array-like
        Design weights, non-negative.
    quantile : float, default 0.99
        Upper percentile at which to cut, in (0, 1].

    Returns
    -------
    RichResult
        ``estimate`` (cut point), ``weights`` (truncated), ``rescaled``
        (truncated then multiplied back to the original total),
        ``n_trimmed``, ``mass_removed``, ``sumw``, ``sumw_trimmed``,
        ``cv_before``, ``cv_after``, ``n``, ``method``.

    References
    ----------
    Potter, F. J. (1990).  A study of procedures to identify and trim
    extreme sampling weights.  Proceedings of the Section on Survey
    Research Methods, American Statistical Association, 225-230.
    """
    w = C.vec(weights)
    n = len(w)
    if n == 0:
        raise ValueError("trim_weights: weights is empty")
    for v in w:
        if v < 0.0:
            raise ValueError("trim_weights: weights must be non-negative")
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("trim_weights: quantile must lie in (0, 1]")
    cut = core.quantile7(w, q)
    tw = [v if v <= cut else cut for v in w]
    ntr = sum(1 for i in range(n) if w[i] > cut)
    s0 = sum(w)
    s1 = sum(tw)
    resc = [v * (s0 / s1) for v in tw] if s1 > 0.0 else list(tw)
    return RichResult(payload={
        "estimate": float(cut), "weights": tw, "rescaled": resc,
        "n_trimmed": int(ntr), "mass_removed": float(s0 - s1),
        "sumw": float(s0), "sumw_trimmed": float(s1),
        "cv_before": _cv(w), "cv_after": _cv(tw), "n": n,
        "method": "weight truncation at the type-7 q-th percentile [Potter 1990]"})


def _cv(w):
    n = len(w)
    m = sum(w) / n
    if m == 0.0:
        return float("nan")
    v = sum((x - m) ** 2 for x in w) / (n - 1) if n > 1 else 0.0
    return (v ** 0.5) / m


# CANONICAL TEST
# >>> r = trim_weights([1.0, 1.0, 1.0, 100.0], 0.75)
# >>> assert abs(r["estimate"] - 1.0) < 1e-12    # quantile(w, .75, type = 7)
# >>> assert r["n_trimmed"] == 1
# >>> assert abs(sum(r["rescaled"]) - 103.0) < 1e-12
# >>> # q = 1 is the identity
# >>> u = trim_weights([1.0, 1.0, 1.0, 100.0], 1.0)
# >>> assert u["n_trimmed"] == 0 and u["cv_after"] == u["cv_before"]


def cheatsheet():
    return "trmwgt(weights, quantile): truncate weights at their q-th percentile."
