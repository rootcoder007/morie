# morie.fn -- function file (rootcoder007/morie)
"""Truncated product of treatment and censoring inverse-probability weights."""

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["truncated_combined_weights"]


def truncated_combined_weights(sw_A, sw_C, quantile=0.99):
    """Combine treatment and censoring weights, then truncate the product.

    A marginal structural model fitted under both non-random treatment
    and non-random censoring uses the product of the two stabilized
    weights.  The product has a much heavier right tail than either
    factor, and Cole and Hernan's advice is to truncate it: accept a
    little bias in exchange for a large variance reduction.  The cut
    point is the type-7 percentile of the product itself.

    Formula: ``sw_i = min(sw_A_i * sw_C_i, Q_q(sw_A * sw_C))``.

    Parameters
    ----------
    sw_A : array-like
        Stabilized treatment weights, positive.
    sw_C : array-like
        Stabilized censoring weights, positive, same length.
    quantile : float, default 0.99
        Upper percentile at which to truncate, in (0, 1].

    Returns
    -------
    RichResult
        ``estimate`` (mean truncated weight, which should sit near 1 for
        correctly stabilized weights), ``weights``, ``cut``,
        ``n_truncated``, ``max_before``, ``max_after``, ``sd``,
        ``mean_untruncated``, ``n``, ``method``.

    References
    ----------
    Cole, S. R. & Hernan, M. A. (2008).  Constructing inverse probability
    weights for marginal structural models.  American Journal of
    Epidemiology 168(6):656-664.  <https://doi.org/10.1093/aje/kwn164>
    """
    a = C.vec(sw_A)
    c = C.vec(sw_C)
    n = len(a)
    if n == 0:
        raise ValueError("truncated_combined_weights: sw_A is empty")
    if len(c) != n:
        raise ValueError("truncated_combined_weights: sw_A and sw_C differ in length")
    for i in range(n):
        if a[i] <= 0.0 or c[i] <= 0.0:
            raise ValueError("truncated_combined_weights: weights must be positive")
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("truncated_combined_weights: quantile must lie in (0, 1]")
    prod = [a[i] * c[i] for i in range(n)]
    cut = core.quantile7(prod, q)
    tw = [v if v <= cut else cut for v in prod]
    m0 = sum(prod) / n
    m1 = sum(tw) / n
    v1 = sum((v - m1) ** 2 for v in tw) / (n - 1) if n > 1 else 0.0
    return RichResult(payload={
        "estimate": float(m1), "weights": tw, "cut": float(cut),
        "n_truncated": int(sum(1 for v in prod if v > cut)),
        "max_before": float(max(prod)), "max_after": float(max(tw)),
        "sd": float(v1 ** 0.5), "mean_untruncated": float(m0), "n": n,
        "method": "sw = min(sw_A sw_C, q-th percentile) [Cole & Hernan 2008]"})


# CANONICAL TEST
# >>> r = truncated_combined_weights([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], 0.99)
# >>> assert abs(r["estimate"] - 1.0) < 1e-12    # stabilized weights average 1
# >>> assert r["n_truncated"] == 0
# >>> # q = 1 never truncates
# >>> u = truncated_combined_weights([1.0, 2.0, 30.0], [1.0, 1.0, 1.0], 1.0)
# >>> assert u["n_truncated"] == 0 and abs(u["estimate"] - u["mean_untruncated"]) < 1e-12


def cheatsheet():
    return "trcwgt(sw_A, sw_C, quantile): truncate the product of IP weights."
