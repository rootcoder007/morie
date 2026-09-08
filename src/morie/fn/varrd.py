# morie.fn -- function file (rootcoder007/morie)
"""Variance reduction criterion for regression tree splitting.

MVSML (2022) sec. 15.4.1 "Splitting Rules" pp.641-642, read from the
chapter-15 split PDF.  The book states the least squares criterion in
its weighted sum-of-squares form,

    SSE = SSE_L Omega_L + SSE_R Omega_R,  Omega_L = n_L / n,

and cites Breiman, Friedman, Olshen & Stone (1984), Classification and
Regression Trees, ch.8.4 for the criterion itself.  The equivalent
variance-reduction form the stub docstring carried,

    Delta_Var = Var(t) - (n_L/n) Var(t_L) - (n_R/n) Var(t_R),

is the CART impurity decrease.  Both are returned: they rank splits
identically, but they are not the same number, so neither is quietly
substituted for the other.
"""

from ._richresult import RichResult, with_describe_pointer

__all__ = ["variance_reduction_split"]


def _pvar(v):
    """Population variance, the node impurity used by CART."""
    n = len(v)
    if n == 0:
        return 0.0
    m = sum(v) / n
    return sum((t - m) ** 2 for t in v) / n


def _sse(v):
    n = len(v)
    if n == 0:
        return 0.0
    m = sum(v) / n
    return sum((t - m) ** 2 for t in v)


def variance_reduction_split(y, split_idx):
    """Impurity decrease of a binary split of a regression node.

    ``split_idx`` selects the left child.  It may be a boolean mask of
    the same length as ``y``, or a sequence of integer positions.

    Returns both the CART variance reduction

        Delta_Var = Var(t) - (n_L/n) Var(t_L) - (n_R/n) Var(t_R)

    with population variances, and the book's weighted criterion
    SSE = SSE_L (n_L/n) + SSE_R (n_R/n) of p.642, which a tree
    minimizes.  A split that puts every observation on one side is
    degenerate and yields a reduction of zero.

    Parameters
    ----------
    y : (n,) array-like of responses at the node.
    split_idx : boolean mask or integer positions of the left child.

    Returns
    -------
    RichResult with keys estimate (the variance reduction), delta_var,
    sse_weighted, sse_left, sse_right, var_parent, var_left,
    var_right, n_left, n_right, omega_left, omega_right, method.

    References
    ----------
    MVSML (2022) sec. 15.4.1 pp.641-642; Breiman et al. (1984) ch.8.4.
    """
    v = [float(t) for t in y]
    n = len(v)
    idx = list(split_idx)
    if len(idx) == n and all(isinstance(t, bool) or t in (0, 1) for t in idx):
        left = [v[i] for i in range(n) if idx[i]]
        right = [v[i] for i in range(n) if not idx[i]]
    else:
        s = set(int(t) for t in idx)
        left = [v[i] for i in range(n) if i in s]
        right = [v[i] for i in range(n) if i not in s]
    nl = len(left)
    nr = len(right)
    wl = nl / n
    wr = nr / n
    vp = _pvar(v)
    vl = _pvar(left)
    vr = _pvar(right)
    dv = vp - wl * vl - wr * vr
    return with_describe_pointer(RichResult(payload={
        "estimate": float(dv), "delta_var": float(dv),
        "sse_weighted": float(_sse(left) * wl + _sse(right) * wr),
        "sse_left": float(_sse(left)), "sse_right": float(_sse(right)),
        "var_parent": float(vp), "var_left": float(vl),
        "var_right": float(vr), "n_left": nl, "n_right": nr,
        "omega_left": float(wl), "omega_right": float(wr),
        "method": "variance reduction split (MVSML 2022 sec. 15.4.1)",
    }), "varrd")


def cheatsheet():
    return "varrd: Variance reduction criterion for regression tree splitting"


# compact alias per ledger/NAMING.md
varreduce = variance_reduction_split
