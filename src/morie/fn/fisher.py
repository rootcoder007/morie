# morie.fn -- function file (rootcoder007/morie)
"""Fisher's exact test for a 2x2 contingency table."""

import math
from typing import Union

from . import _array_core as np
from . import _stats_core as stats


def _log_choose(a: int, b: int) -> float:
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def _conditional_or(a: int, b: int, c: int, d: int, conf_level: float = 0.95) -> tuple:
    """Conditional MLE and exact interval of the odds ratio (Cornfield 1956), as R's
    ``fisher.test`` and rmorie's ``.morie_fisher_conditional``: the noncentral hypergeometric
    mean and tails solved by bisection to machine precision."""
    x, m, n, k = a, a + c, b + d, a + b
    lo, hi = max(0, k - n), min(k, m)
    s = list(range(lo, hi + 1))
    ld = [_log_choose(m, v) + _log_choose(n, k - v) - _log_choose(m + n, k) for v in s]

    def dn(p):
        lp = math.log(p)
        e = [u + lp * v for u, v in zip(ld, s)]
        top = max(e)
        w = [math.exp(t - top) for t in e]
        tot = math.fsum(w)
        return [t / tot for t in w]

    def mn(p):
        if p == 0:
            return lo
        if math.isinf(p):
            return hi
        return math.fsum(v * w for v, w in zip(s, dn(p)))

    def pn(q, p, upper=False):
        # fisher.test's pnhyper: the degenerate ncp = 0 / Inf distributions sit at lo / hi
        if p == 0:
            return float(q <= lo if upper else q >= lo)
        if math.isinf(p):
            return float(q <= hi if upper else q >= hi)
        return math.fsum(w for v, w in zip(s, dn(p)) if (v >= q if upper else v <= q))

    def bis(f, lo_, hi_):
        fa = f(lo_)
        for _ in range(300):
            mid = (lo_ + hi_) / 2
            fm = f(mid)
            if (fm > 0) == (fa > 0):
                lo_, fa = mid, fm
            else:
                hi_ = mid
            if hi_ - lo_ <= 1e-16:
                break
        return (lo_ + hi_) / 2

    eps = 2.220446049250313e-16
    if x == lo:
        mle = 0.0
    elif x == hi:
        mle = math.inf
    else:
        mu = mn(1)
        if mu > x:
            mle = bis(lambda t: mn(t) - x, 0, 1)
        elif mu < x:
            mle = 1 / bis(lambda t: mn(1 / t) - x, eps, 1)
        else:
            mle = 1.0
    alpha = (1 - conf_level) / 2
    if x == hi:
        up = math.inf
    else:
        p = pn(x, 1)
        if p < alpha:
            up = bis(lambda t: pn(x, t) - alpha, 0, 1)
        elif p > alpha:
            up = 1 / bis(lambda t: pn(x, 1 / t) - alpha, eps, 1)
        else:
            up = 1.0
    if x == lo:
        low = 0.0
    else:
        p = pn(x, 1, True)
        if p > alpha:
            low = bis(lambda t: pn(x, t, True) - alpha, 0, 1)
        elif p < alpha:
            low = 1 / bis(lambda t: pn(x, 1 / t, True) - alpha, eps, 1)
        else:
            low = 1.0
    return mle, low, up


def fisher_exact_test(table_2x2: Union[list, np.ndarray]) -> dict:
    """
    Fisher's exact test for a 2x2 contingency table.

    Appropriate when expected cell counts are small (< 5) and the chi-square
    approximation is unreliable. The odds ratio is the conditional maximum-likelihood
    estimate with its exact 95% interval, as R's ``fisher.test`` and rmorie's
    ``morie_fisher_exact_test`` report it (not the sample ratio ad/bc).

    :param table_2x2: A 2x2 array-like [[a, b], [c, d]].
    :return: dict with keys ``odds_ratio``, ``ci``, ``p_value``.
    :raises ValueError: If the table is not 2x2 or contains negative values.

    Examples
    --------
    >>> r = fisher_exact_test([[8, 2], [1, 5]])
    >>> round(r["odds_ratio"], 6), round(r["p_value"], 9)
    (15.46582, 0.034965035)

    References
    ----------
    Fisher, R. A. (1935). The logic of inductive inference. Journal of the Royal
        Statistical Society, 98(1), 39-82.
    Cornfield, J. (1956). A statistical problem arising from retrospective studies.
        Proceedings of the Third Berkeley Symposium, 4, 135-148.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must have shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Contingency table entries must be non-negative.")
    _, p_val = stats.fisher_exact(tbl.astype(int))
    (a, b), (c, d) = [[int(round(float(v))) for v in row] for row in tbl.tolist()]
    odds_ratio, low, up = _conditional_or(a, b, c, d)
    return {
        "odds_ratio": float(odds_ratio),
        "ci": [float(low), float(up)],
        "p_value": float(p_val),
        "method": "Fisher's exact test",
    }


fisher = fisher_exact_test


def cheatsheet() -> str:
    return "fisher_exact_test({}) -> Fisher's exact test for a 2x2 contingency table."
