# morie.fn -- function file (rootcoder007/morie)
"""Linear min-max (intersection) bound."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_linear_min_max"]


def bound_linear_min_max(theta, moments):
    """Intersection bounds from many candidate lower and upper bounds.

    When several inequalities each bound the same parameter, the binding
    one is the largest lower and the smallest upper.  Taking a min of
    noisy estimates biases it downward, so the plug-in upper bound is too
    tight; the precision correction pushes each candidate out by its own
    standard error before the min is taken.  The correction reported here
    is the Bonferroni one, which is exact at a single candidate (the
    multiplier is then zero) and conservative beyond it.

    Formula: ``[max_k m_k, min_j m_j]`` plug-in, and
    ``[max_k (m_k - z_K s_k / sqrt(n)), min_j (m_j + z_J s_j / sqrt(n))]``
    with ``z_J = Phi^{-1}(1 - 1/(2J))`` for the half-median-unbiased
    version.

    Parameters
    ----------
    theta : array-like, shape (n, K)
        Observation-level estimates of ``K`` candidate lower bounds.
    moments : array-like, shape (n, J)
        Observation-level estimates of ``J`` candidate upper bounds.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width``, ``lower_pc``, ``upper_pc``,
        ``width_pc``, ``K``, ``J``, ``n``.

    References
    ----------
    Chernozhukov, V., Lee, S. & Rosen, A. M. (2013).  Intersection
    bounds: estimation and inference.  Econometrica 81(2), 667-737.
    doi:10.3982/ECTA8718.  The half-median-unbiased criterion the
    correction targets is equation (4.9) of Molinari, F. (2021),
    Handbook of Econometrics 7A (arXiv:2004.11751 p. 96); the multiplier
    used here is Bonferroni rather than the paper's bootstrap, and is
    labelled as such.
    """
    L = C.mat(theta)
    U = C.mat(moments)
    n = len(L)
    if n < 2:
        raise ValueError("bound_linear_min_max: need at least two observations")
    if len(U) != n:
        raise ValueError("bound_linear_min_max: theta and moments must have the same number of rows")
    K = len(L[0])
    J = len(U[0])
    rn = n ** 0.5
    zK = C.qnorm(1.0 - 0.5 / K)
    zJ = C.qnorm(1.0 - 0.5 / J)
    lo = None
    lo_pc = None
    for k in range(K):
        col = [r[k] for r in L]
        m = C.mean(col)
        s = C.sd(col)
        if lo is None or m > lo:
            lo = m
        v = m - zK * s / rn
        if lo_pc is None or v > lo_pc:
            lo_pc = v
    hi = None
    hi_pc = None
    for j in range(J):
        col = [r[j] for r in U]
        m = C.mean(col)
        s = C.sd(col)
        if hi is None or m < hi:
            hi = m
        v = m + zJ * s / rn
        if hi_pc is None or v < hi_pc:
            hi_pc = v
    return RichResult(payload={
        "lower": lo, "upper": hi, "width": hi - lo,
        "lower_pc": lo_pc, "upper_pc": hi_pc, "width_pc": hi_pc - lo_pc,
        "K": K, "J": J, "n": n,
        "method": "Linear min-max bound"})


def cheatsheet():
    return "bndlmm: Linear min-max bound"
