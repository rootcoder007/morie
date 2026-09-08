# morie.fn -- function file (rootcoder007/morie)
"""Optimal regression-tree treatment regime."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["optimal_tree_regime"]


def optimal_tree_regime(y, A, W, pi=None, max_depth=2, min_leaf=1):
    """Interpretable treatment regime by exhaustive tree search.

    Laber & Zhao's argument is that a regime is only useful if a
    clinician can read it, and that the usual two-step approach --
    regress the outcome, then take the argmax -- optimizes the wrong
    thing: a small error in the outcome model at a point where the two
    arms are close flips the recommendation.  Their tree instead
    maximizes the ESTIMATED VALUE directly,

        V(d) = sum_i Y_i 1{A_i = d(W_i)} / pi_i
               / sum_i 1{A_i = d(W_i)} / pi_i,

    over axis-aligned splits.  This implementation enumerates every
    split of every covariate at every observed value and recurses --
    exhaustive, so there is nothing to seed and the tree is the same
    in both language arms bit for bit.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome, larger is better.
    A : array-like, shape (n,)
        Observed binary treatment, 0/1.
    W : array-like, shape (n, p)
        Covariates.
    pi : array-like or None
        Propensity ``P(A = A_i | W_i)``; the marginal randomization
        probability if ``None``.
    max_depth : int, default 2
        Maximum tree depth; 0 gives a single constant recommendation.
    min_leaf : int, default 1
        Minimum observations in a leaf.

    Returns
    -------
    RichResult
        ``estimate`` (value of the learned regime), ``value``,
        ``value_all_treated``, ``value_all_control``, ``rule``
        (recommended treatment per unit), ``split_var``,
        ``split_point`` (root split, ``-1``/NaN if the root is a leaf),
        ``n_leaves``, ``depth``, ``n``.

    References
    ----------
    Laber, E. B. & Zhao, Y.-Q. (2015).  Tree-based methods for
    individualized treatment regimes.  Biometrika, 102(3), 501--514.
    doi:10.1093/biomet/asv028
    """
    yv = C.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("optimal_tree_regime: y is empty")
    Av = C.vec(A)
    if len(Av) != n:
        raise ValueError("optimal_tree_regime: y and A have different lengths")
    if any(v not in (0.0, 1.0) for v in Av):
        raise ValueError("optimal_tree_regime: A must be binary 0/1")
    Wm = C.mat(W)
    if len(Wm) != n:
        raise ValueError("optimal_tree_regime: W and y have different lengths")
    p = len(Wm[0])
    md = int(max_depth)
    if md < 0:
        raise ValueError("optimal_tree_regime: max_depth must be non-negative")
    ml = int(min_leaf)
    if ml < 1:
        raise ValueError("optimal_tree_regime: min_leaf must be at least 1")
    if pi is None:
        pt = sum(Av) / n
        if pt <= 0.0 or pt >= 1.0:
            raise ValueError("optimal_tree_regime: both treatments must be observed")
        pv = [pt if Av[i] == 1.0 else 1.0 - pt for i in range(n)]
    else:
        pv = C.vec(pi)
        if len(pv) != n:
            raise ValueError("optimal_tree_regime: pi and y have different lengths")
        if any(v <= 0.0 or v > 1.0 for v in pv):
            raise ValueError("optimal_tree_regime: pi must lie in (0, 1]")

    def leaf_score(idx, a):
        """IPW numerator and denominator for recommending ``a`` on ``idx``."""
        num = 0.0
        den = 0.0
        for i in idx:
            if Av[i] == a:
                num += yv[i] / pv[i]
                den += 1.0 / pv[i]
        return num, den

    def best_const(idx):
        n1, d1 = leaf_score(idx, 1.0)
        n0, d0 = leaf_score(idx, 0.0)
        v1 = n1 / d1 if d1 > 0.0 else float("-inf")
        v0 = n0 / d0 if d0 > 0.0 else float("-inf")
        if v1 > v0:
            return 1.0, n1, d1
        return 0.0, n0, d0

    rule = [0.0] * n
    root = {"var": -1, "point": float("nan")}
    depth_used = [0]

    def build(idx, depth, record_root):
        a, num, den = best_const(idx)
        base = (num, den)
        best = None
        if depth < md and len(idx) >= 2 * ml:
            for j in range(p):
                for i in idx:
                    thr = Wm[i][j]
                    left = [k for k in idx if Wm[k][j] <= thr]
                    right = [k for k in idx if Wm[k][j] > thr]
                    if len(left) < ml or len(right) < ml:
                        continue
                    al, nl, dl = best_const(left)
                    ar, nr, dr = best_const(right)
                    if dl <= 0.0 or dr <= 0.0:
                        continue
                    v = (nl + nr) / (dl + dr)
                    if best is None or v > best[0]:
                        best = (v, j, thr, left, right)
        v_leaf = base[0] / base[1] if base[1] > 0.0 else float("-inf")
        if best is not None and best[0] > v_leaf:
            _, j, thr, left, right = best
            if record_root:
                root["var"] = j
                root["point"] = thr
            if depth + 1 > depth_used[0]:
                depth_used[0] = depth + 1
            nl_ = build(left, depth + 1, False)
            nr_ = build(right, depth + 1, False)
            return nl_ + nr_
        for i in idx:
            rule[i] = a
        return 1

    n_leaves = build(list(range(n)), 0, True)

    def value(rec):
        num = 0.0
        den = 0.0
        for i in range(n):
            if rec[i] == Av[i]:
                num += yv[i] / pv[i]
                den += 1.0 / pv[i]
        return num / den if den > 0.0 else float("nan")

    return RichResult(payload={
        "estimate": value(rule), "value": value(rule),
        "value_all_treated": value([1.0] * n),
        "value_all_control": value([0.0] * n), "rule": rule,
        "split_var": float(root["var"]), "split_point": root["point"],
        "n_leaves": float(n_leaves), "depth": float(depth_used[0]), "n": n,
        "method": "Value-search treatment regime tree (Laber & Zhao 2015)"})


def cheatsheet():
    return "opttre: Optimal treatment regime by exhaustive tree search"


optimaltreeregime = optimal_tree_regime
