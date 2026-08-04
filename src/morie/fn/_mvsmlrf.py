# morie.fn -- shared core (rootcoder007/morie)
"""Deterministic random forest core for the Chapter 15 modules.

Shared by rfmlt, rfmdi and rfpmi.  The algorithm is the one printed in
Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*, Springer,
volume [Pages 633-681], Chapter 15, Section 15.4, pp. 639-640, read as
rendered page images:

    For b = 1, ..., B bootstrap samples {y_b, X_b}
    Step 1. From the training data set, draw bootstrap samples of size
            N_train.
    Step 2. With the bootstrapped data, grow a random forest tree T_b
            with the specific splitting criterion, by recursively
            repeating the following steps for each terminal node of the
            tree, until the minimum node size is reached.
            (a) Randomly draw mtry out of the p independent variables;
                mtry is a user-specified parameter and should be less or
                equal to p.
            (b) Pick the best independent variable among the mtry IVs.
            (c) Split the node into two child nodes.  The split ends when
                a stopping criterion is reached, for instance, when a
                node has less than a predetermined number of
                observations.  No pruning is performed.
    Step 3. The ensemble of trees is obtained {T_b}_1^B.

and the prediction is yhat_i = (1/B) sum_b T_b(x_i) for a continuous
response.  Page 643 gives the defaults this core uses: mtry = p/3 for
regression, "values are always rounded up", and nodesize 5 for
regression.  The splitting rule is the weighted mean squared error, "the
least square criterion", which p. 643 attributes to Breiman, Friedman,
Olshen and Stone (1984), Chapter 8.4.

DETERMINISM.  The book's steps 1 and 2(a) both say "randomly", and a
random forest whose two arms draw different samples cannot be compared at
1e-9.  Both are therefore replaced by the van der Corput low-discrepancy
sequence already used elsewhere in this package, which is deterministic
and identical in both arms:

  * the bootstrap of step 1 is drawn by the Numerical Recipes 32-bit
    linear congruential generator x <- (1664525 x + 1013904223) mod 2^32,
    seeded from the tree index, and NOT by a low-discrepancy sequence.
    That distinction is load-bearing.  A van der Corput sweep was tried
    first and is wrong here: being low-discrepancy it visits almost every
    index exactly once, so the out-of-bag set collapsed to 2 rows out of
    40 instead of the 36.8% that p. 640 states ("each tree makes use of
    around two-thirds (63.2%) of the observations ... the remaining
    observations are referred to as Out-Of-Bag").  Permutation importance
    is computed on that set, so an empty one would have made rfpmi
    meaningless while still passing parity.  The LCG draws with genuine
    replacement and reproduces the 63.2% figure;
  * the mtry candidate variables at the s-th split of tree b start at
    offset floor(vdc(b*4096 + s + 1, base 3) * p) and run consecutively
    modulo p.

Ties in the split search go to the lowest variable index and then to the
lowest threshold, so the tree is fully determined by the data.
"""

from __future__ import annotations

from . import _s03core as core

__all__: list[str] = []


_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 4294967296


def bootstrap_rows(b, n):
    """Step 1: a resample of size n drawn with replacement, deterministically.

    The generator is the Numerical Recipes 32-bit LCG seeded from the tree
    index.  Every quantity here is an exact integer below 2^53, so the R
    mirror reproduces it bit for bit in double precision.
    """
    x = (b + 1) * 2654435761 % _LCG_M
    rows = []
    for _ in range(n):
        x = (_LCG_A * x + _LCG_C) % _LCG_M
        rows.append(int(x / _LCG_M * n))
    return [0 if i < 0 else (n - 1 if i > n - 1 else i) for i in rows]


def candidate_vars(b, s, p, mtry):
    """Deterministic stand-in for step 2(a)'s draw of mtry variables."""
    off = int(core.vdc(b * 4096 + s + 1, 3) * p)
    return [(off + j) % p for j in range(mtry)]


def default_mtry(p, kind="regression"):
    """p/3 for regression, sqrt(p) otherwise; p. 643, always rounded up."""
    if kind == "regression":
        m = -((-p) // 3)
    else:
        m = 1
        while m * m < p:
            m += 1
    return max(1, min(p, m))


def _node_stats(Y, rows, cols):
    """Sums and sum of squares of every response column over `rows`."""
    s = [0.0] * cols
    for i in rows:
        for j in range(cols):
            s[j] += Y[i][j]
    return s


def _mss_gain(Y, left, right, q):
    """The (15.6) objective: sum_j [ (sum_L y*)^2/n_L + (sum_R y*)^2/n_R ].

    Maximised, not minimised -- see the erratum recorded in rfmlt.
    """
    nL, nR = len(left), len(right)
    if nL == 0 or nR == 0:
        return None
    sL = _node_stats(Y, left, q)
    sR = _node_stats(Y, right, q)
    g = 0.0
    for j in range(q):
        g += sL[j] * sL[j] / nL + sR[j] * sR[j] / nR
    return g


def _impurity(Y, rows, q):
    """Within-node sum of squares, the least-square criterion of p. 643."""
    n = len(rows)
    if n == 0:
        return 0.0
    s = _node_stats(Y, rows, q)
    tot = 0.0
    for i in rows:
        for j in range(q):
            tot += Y[i][j] * Y[i][j]
    for j in range(q):
        tot -= s[j] * s[j] / n
    return tot


def best_split(X, Y, rows, cand, q):
    """Exhaustive search over midpoints, ties to the lowest index/threshold."""
    best = None
    for v in cand:
        vals = sorted(set(X[i][v] for i in rows))
        for a in range(len(vals) - 1):
            thr = 0.5 * (vals[a] + vals[a + 1])
            left = [i for i in rows if X[i][v] <= thr]
            right = [i for i in rows if X[i][v] > thr]
            g = _mss_gain(Y, left, right, q)
            if g is None:
                continue
            if best is None or g > best[0] + 1e-15:
                best = (g, v, thr, left, right)
    return best


def grow_tree(X, Y, rows, b, nodesize, mtry, q, counter, nodes):
    """Recursively grow one tree; returns the index of its root node.

    A node is (var, thr, left, right, value, n, impurity_drop).  A leaf has
    var = -1 and value the column means over its rows.
    """
    n = len(rows)
    idx = len(nodes)
    nodes.append(None)
    mean = [0.0] * q
    if n:
        s = _node_stats(Y, rows, q)
        mean = [v / n for v in s]
    if n < 2 * nodesize or n < 2:
        nodes[idx] = (-1, 0.0, -1, -1, mean, n, 0.0, -1)
        return idx
    p = len(X[0])
    cand = candidate_vars(b, counter[0], p, mtry)
    counter[0] += 1
    sp = best_split(X, Y, rows, cand, q)
    if sp is None or len(sp[3]) < nodesize or len(sp[4]) < nodesize:
        nodes[idx] = (-1, 0.0, -1, -1, mean, n, 0.0, -1)
        return idx
    _, var, thr, left, right = sp
    # impurity decrease, in the raw sum-of-squares scale, for the MDI sum
    drop = _impurity(Y, rows, q) - _impurity(Y, left, q) - _impurity(Y, right, q)
    li = grow_tree(X, Y, left, b, nodesize, mtry, q, counter, nodes)
    ri = grow_tree(X, Y, right, b, nodesize, mtry, q, counter, nodes)
    nodes[idx] = (var, thr, li, ri, mean, n, drop, var)
    return idx


def predict_tree(nodes, root, x):
    i = root
    while nodes[i][0] != -1:
        i = nodes[i][2] if x[nodes[i][0]] <= nodes[i][1] else nodes[i][3]
    return nodes[i][4]


def build_forest(X, Y, n_trees, nodesize, mtry, q):
    """Grow the ensemble; returns (trees, oob) with oob[b] the held-out rows."""
    n = len(X)
    trees = []
    oob = []
    for b in range(n_trees):
        rows = bootstrap_rows(b, n)
        inbag = set(rows)
        nodes = []
        root = grow_tree(X, Y, rows, b, nodesize, mtry, q, [0], nodes)
        trees.append((nodes, root))
        oob.append([i for i in range(n) if i not in inbag])
    return trees, oob


def forest_predict(trees, X_new, q):
    """yhat_i = (1/B) sum_b T_b(x_i), the p. 640 aggregation."""
    B = len(trees)
    out = []
    for x in X_new:
        acc = [0.0] * q
        for nodes, root in trees:
            v = predict_tree(nodes, root, x)
            for j in range(q):
                acc[j] += v[j]
        out.append([a / B for a in acc])
    return out


def check_xy(X, Y):
    """Shared validation for the three Chapter 15 modules."""
    n = len(X)
    if n == 0:
        raise ValueError("random forest: X is empty")
    p = len(X[0])
    if p == 0:
        raise ValueError("random forest: X has no columns")
    for r in X:
        if len(r) != p:
            raise ValueError("random forest: X rows have unequal lengths")
    if len(Y) != n:
        raise ValueError("random forest: Y has a different number of rows than X")
    q = len(Y[0])
    if q == 0:
        raise ValueError("random forest: Y has no columns")
    for r in Y:
        if len(r) != q:
            raise ValueError("random forest: Y rows have unequal lengths")
    return n, p, q


def standardize(Y, n, q):
    """The p. 656 requirement: responses standardized before splitting."""
    out = [[0.0] * q for _ in range(n)]
    for j in range(q):
        m = sum(Y[i][j] for i in range(n)) / n
        v = sum((Y[i][j] - m) ** 2 for i in range(n)) / n
        sd = v ** 0.5
        for i in range(n):
            out[i][j] = (Y[i][j] - m) / sd if sd > 0.0 else 0.0
    return out
