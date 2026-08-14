# morie.fn -- function file (rootcoder007/morie)
r"""Honest random forests, with valid pointwise inference.

An ordinary regression tree chooses its splits and estimates its leaf
means from the same observations. That is what makes a forest a good
predictor and a bad estimator: the split is placed where the noise
happened to fall, and the leaf mean then inherits that noise as bias.
No amount of averaging over trees removes it, because every tree makes
the same kind of mistake.

**Honesty is a structural fix, not a tuning parameter.** Definition 2:
a tree is honest if it does not use the responses it will later average.
Two constructions achieve it:

``"double-sample"``
    Procedure 1. Draw a subsample of size :math:`s`, split it into
    disjoint :math:`I` and :math:`J`. Place splits using :math:`J`'s
    responses and :math:`I`'s features -- never :math:`I`'s responses --
    then estimate every leaf from :math:`I` alone.

``"propensity"``
    Procedure 2. Ignore the outcome entirely when splitting and split
    on the treatment instead, which buys honesty without spending half
    the sample on it.

The anchor tests honesty the only way that can actually fail: it
*permutes the I-sample responses* and checks the tree structure does
not move. A forest that quietly used them would shift, and no accuracy
comparison would reveal it.

**Two more conditions carry the theory.** A tree is *random-split*
(Definition 3) if at every step each feature has probability at least
:math:`\pi/d` of being the one split on, which is what forces the
leaves to shrink in every dimension rather than only in the few
features that happen to predict well. It is
:math:`\alpha`-*regular* (Definition 4) if every split leaves at least
a fraction :math:`\alpha` on each side. Both are enforced here rather
than assumed, and both are checked.

**The variance estimate.** The infinitesimal jackknife, eq. (8):

.. math:: \hat V_{IJ}(x) = \frac{n-1}{n}\Big(\frac{n}{n-s}\Big)^2
          \sum_{i=1}^{n} \mathrm{Cov}_*\big[\hat\mu^*_b(x),
          N^*_{ib}\big]^2,

the covariance taken across trees between a tree's prediction and
whether observation :math:`i` was in its subsample. The factor
:math:`n(n-1)/(n-s)^2` is a finite-sample correction for subsampling
*without* replacement; drop it and the intervals are too narrow at any
realistic ratio of :math:`s` to :math:`n`. The anchor reports coverage
with and without it.

References
----------
Wager, S. & Athey, S. (2018) "Estimation and Inference of Heterogeneous
Treatment Effects using Random Forests", *Journal of the American
Statistical Association* 113(523), 1228-1242,
doi:10.1080/01621459.2017.1319839, arXiv:1510.04342. Procedures 1-2,
Definitions 1-5, eq. (8), Theorem 1.

Athey, S. & Imbens, G. (2016) "Recursive partitioning for heterogeneous
causal effects", *Proceedings of the National Academy of Sciences*
113(27), 7353-7360, doi:10.1073/pnas.1510489113. The honest causal tree
and its splitting criterion.

Wager, S., Hastie, T. & Efron, B. (2014) "Confidence Intervals for
Random Forests: The Jackknife and the Infinitesimal Jackknife",
*Journal of Machine Learning Research* 15(1), 1625-1651. The variance
estimator eq. (8) applies.

Efron, B. (2014) "Estimation and Accuracy after Model Selection",
*Journal of the American Statistical Association* 109(507), 991-1007,
doi:10.1080/01621459.2013.823775. The infinitesimal jackknife it is
built on.

Breiman, L. (2001) "Random Forests", *Machine Learning* 45(1), 5-32,
doi:10.1023/A:1010933404324. The forest this makes honest.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["honest_forest", "honest_tree", "tree_predict",
           "infinitesimal_jackknife", "leaf_of", "forest_weights",
           "grow_forest"]

_KINDS = ("double-sample", "propensity", "adaptive")
_EPS = 1e-12


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _best_split(X, y, rows, feats, min_leaf, alpha):
    """The CART split on `rows`, restricted to `feats`.

    Only rows whose responses the caller is allowed to look at are
    passed in -- honesty is enforced by the caller choosing which rows
    reach here, not by anything this function does.
    """
    n = len(rows)
    if n < 2 * min_leaf:
        return None
    base = _mean([y[i] for i in rows])
    tot = sum((y[i] - base) ** 2 for i in rows)
    best = None
    floor = max(min_leaf, int(math.ceil(alpha * n)))
    for f in feats:
        order = sorted(rows, key=lambda i: X[i][f])
        vals = [X[i][f] for i in order]
        ys = [y[i] for i in order]
        csum, csq = 0.0, 0.0
        for t in range(n - 1):
            csum += ys[t]
            csq += ys[t] * ys[t]
            left, right = t + 1, n - t - 1
            if left < floor or right < floor:
                continue
            if vals[t] == vals[t + 1]:
                continue
            rsum = sum(ys) - csum
            sse = (csq - csum * csum / left)
            sse += (sum(v * v for v in ys) - csq
                    - rsum * rsum / right)
            gain = tot - sse
            if best is None or gain > best[0]:
                best = (gain, f, 0.5 * (vals[t] + vals[t + 1]))
    return best


def honest_tree(X, y, W=None, kind="double-sample", min_leaf=5,
                alpha=0.05, pi=0.5, max_depth=12, seed=0,
                subsample=None):
    r"""One tree of Procedure 1 or 2.

    Returns the tree structure and the index sets it used, so a caller
    -- or an anchor -- can check *which* responses touched the splits.
    """
    if kind not in _KINDS:
        raise ValueError("hntfst: kind must be one of %s, got %r"
                         % (", ".join(_KINDS), kind))
    n = len(y)
    d = len(X[0]) if n and X[0] else 0
    if d == 0:
        raise ValueError("hntfst: no features")
    if not 0.0 < alpha < 0.5:
        raise ValueError("hntfst: alpha must be in (0, 0.5), got %r"
                         % (alpha,))
    if not 0.0 < pi <= 1.0:
        raise ValueError("hntfst: pi must be in (0, 1], got %r" % (pi,))
    rng = np.random.default_rng(seed)
    sub = list(range(n)) if subsample is None else list(subsample)
    s = len(sub)
    if s < 4 * min_leaf:
        raise ValueError("hntfst: subsample of %d is too small for a "
                         "minimum leaf of %d" % (s, min_leaf))

    if kind == "double-sample":
        # Procedure 1 step 1: split the subsample into disjoint I and J
        perm = sorted(sub, key=lambda _i: float(rng.uniform()))
        half = s // 2
        I, J = perm[:half], perm[half:]
    elif kind == "propensity":
        # Procedure 2: splits ignore Y altogether, so no split is
        # needed -- every row estimates and every row splits
        I = J = list(sub)
    else:
        I = J = list(sub)                       # adaptive: not honest

    if kind == "propensity":
        if W is None:
            raise ValueError("hntfst: a propensity tree needs W")
        split_target = [float(v) for v in W]
    else:
        split_target = [float(v) for v in y]

    def grow(rows_J, rows_I, depth):
        node = {"leaf": True, "I": list(rows_I), "J": list(rows_J),
                "value": _mean([y[i] for i in rows_I]) if rows_I
                else 0.0, "n_I": len(rows_I)}
        if depth >= max_depth or len(rows_I) < 2 * min_leaf:
            return node
        # Definition 3: each feature has probability at least pi/d of
        # being available, so leaves shrink in EVERY dimension rather
        # than only in the features that predict well
        feats = [f for f in range(d) if float(rng.uniform()) < max(
            pi, 1.0 / d)]
        if not feats:
            feats = [int(float(rng.uniform()) * d) % d]
        sp = _best_split(X, split_target, rows_J, feats, min_leaf,
                         alpha)
        if sp is None:
            return node
        _, f, thr = sp
        JL = [i for i in rows_J if X[i][f] <= thr]
        JR = [i for i in rows_J if X[i][f] > thr]
        IL = [i for i in rows_I if X[i][f] <= thr]
        IR = [i for i in rows_I if X[i][f] > thr]
        # Definition 4 for the I sample: the leaf must hold enough of
        # the rows that will actually be averaged
        if len(IL) < min_leaf or len(IR) < min_leaf:
            return node
        if not JL or not JR:
            return node
        return {"leaf": False, "feature": f, "threshold": thr,
                "left": grow(JL, IL, depth + 1),
                "right": grow(JR, IR, depth + 1)}

    tree = grow(J, I, 0)
    return tree, {"I": I, "J": J, "kind": kind, "subsample": sub}


def leaf_of(tree, x):
    """Walk a point down to its leaf."""
    node = tree
    path = []
    while not node["leaf"]:
        path.append((node["feature"], node["threshold"]))
        node = (node["left"] if x[node["feature"]] <= node["threshold"]
                else node["right"])
    return node, path


def tree_predict(tree, x):
    """The leaf value, estimated from the I sample alone."""
    return leaf_of(tree, x)[0]["value"]


def infinitesimal_jackknife(preds, in_bag, n, s, correction=True):
    r"""Eq. (8): the IJ variance of a forest prediction at one point.

    ``preds`` is each tree's prediction and ``in_bag[b][i]`` whether
    observation i was in tree b's subsample. The
    :math:`n(n-1)/(n-s)^2` factor corrects for subsampling *without*
    replacement; dropping it gives intervals that are too narrow.
    """
    B = len(preds)
    if B < 2:
        raise ValueError("hntfst: the IJ variance needs at least 2 "
                         "trees, got %d" % B)
    if n <= s:
        raise ValueError("hntfst: need n > s for the IJ correction, "
                         "got n=%d s=%d" % (n, s))
    pbar = _mean(preds)
    total = 0.0
    for i in range(n):
        nbar = _mean([1.0 if in_bag[b][i] else 0.0 for b in range(B)])
        cov = sum((preds[b] - pbar)
                  * ((1.0 if in_bag[b][i] else 0.0) - nbar)
                  for b in range(B)) / B
        total += cov * cov
    if correction:
        total *= (n - 1.0) / n * (float(n) / (n - s)) ** 2
    return total


def honest_forest(X, y, W=None, kind="double-sample", n_trees=200,
                  subsample_frac=0.5, min_leaf=5, alpha=0.05, pi=0.5,
                  max_depth=12, seed=0, at=None, level=0.95,
                  correction=True):
    r"""Definition 1: average honest trees over subsamples of size s.

    Parameters
    ----------
    at : array-like, optional
        Points to predict at. Defaults to the training features.
    kind : {"double-sample", "propensity", "adaptive"}
        ``adaptive`` is the ordinary, NON-honest forest -- splits and
        leaf estimates from the same rows. It is here because the bias
        it carries is the reason honesty exists, and a bias you can
        measure beats one you assert.

    Returns
    -------
    RichResult
        ``estimate`` are the predictions, with ``se`` and ``ci`` from
        the infinitesimal jackknife.
    """
    yv = k.vec(y)
    n = len(yv)
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("hntfst: %d feature rows for %d responses"
                         % (len(Xm), n))
    if n < 16:
        raise ValueError("hntfst: need at least 16 observations, got %d"
                         % n)
    if not 0.0 < subsample_frac < 1.0:
        raise ValueError("hntfst: subsample_frac must be in (0, 1), "
                         "got %r" % (subsample_frac,))
    s = max(4 * min_leaf, int(subsample_frac * n))
    if s >= n:
        raise ValueError("hntfst: the subsample must be smaller than n")
    Q = k.mat(at) if at is not None else Xm
    B = int(n_trees)
    if B < 2:
        raise ValueError("hntfst: need at least 2 trees, got %d" % B)

    rng = np.random.default_rng(seed)
    preds = [[0.0] * len(Q) for _ in range(B)]
    in_bag = [[False] * n for _ in range(B)]
    splits_on = [0] * (len(Xm[0]) if Xm and Xm[0] else 1)
    depths = []
    for b in range(B):
        sub = sorted(range(n),
                     key=lambda _i: float(rng.uniform()))[:s]
        for i in sub:
            in_bag[b][i] = True
        tree, info = honest_tree(Xm, yv, W=W, kind=kind,
                                 min_leaf=min_leaf, alpha=alpha, pi=pi,
                                 max_depth=max_depth, seed=seed * 7919
                                 + b, subsample=sub)
        for q in range(len(Q)):
            preds[b][q] = tree_predict(tree, Q[q])

        def walk(nd, dep):
            if nd["leaf"]:
                depths.append(dep)
                return
            splits_on[nd["feature"]] += 1
            walk(nd["left"], dep + 1)
            walk(nd["right"], dep + 1)

        walk(tree, 0)

    fitted = [_mean([preds[b][q] for b in range(B)])
              for q in range(len(Q))]
    var = [infinitesimal_jackknife([preds[b][q] for b in range(B)],
                                   in_bag, n, s, correction=correction)
           for q in range(len(Q))]
    se = [math.sqrt(max(v, 0.0)) for v in var]
    z = k.qnorm(0.5 + 0.5 * float(level))
    tot_splits = sum(splits_on) or 1
    return RichResult(payload={
        "estimate": fitted, "fitted": fitted, "se": se,
        "ci": [(fitted[q] - z * se[q], fitted[q] + z * se[q])
               for q in range(len(Q))],
        "variance": var, "n": n, "s": s, "n_trees": B,
        "split_counts": splits_on,
        "split_share": [v / tot_splits for v in splits_on],
        "mean_depth": _mean(depths), "kind": kind,
        "honest": kind != "adaptive", "correction": bool(correction),
        "level": float(level),
        "method": "honest random forest, Wager & Athey (2018) "
                  "Procedures 1-2, Definitions 1-5, eq. (8)",
    })


def grow_forest(X, y, W=None, kind="double-sample", n_trees=200,
                subsample_frac=0.5, min_leaf=5, alpha=0.05, pi=0.5,
                max_depth=12, seed=0, clusters=None):
    """Grow the trees once and hand them back, so the callers that need
    the forest's NEIGHBOURHOOD rather than its predictions do not each
    re-grow it.

    ``clusters`` makes the subsample draw whole clusters instead of
    individual rows, which is what makes cluster-robust inference
    possible downstream.
    """
    n = len(y)
    s = max(4 * min_leaf, int(subsample_frac * n))
    if s >= n:
        raise ValueError("hntfst: the subsample must be smaller than n")
    rng = np.random.default_rng(seed)
    if clusters is not None:
        lab = [str(c) for c in clusters]
        if len(lab) != n:
            raise ValueError("hntfst: %d cluster labels for %d rows"
                             % (len(lab), n))
        groups = {}
        for i, c in enumerate(lab):
            groups.setdefault(c, []).append(i)
        keys = sorted(groups)
        n_keep = max(2, int(subsample_frac * len(keys)))
    trees, bags = [], []
    for b in range(int(n_trees)):
        if clusters is not None:
            pick = sorted(keys,
                          key=lambda _c: float(rng.uniform()))[:n_keep]
            sub = [i for c in pick for i in groups[c]]
        else:
            sub = sorted(range(n),
                         key=lambda _i: float(rng.uniform()))[:s]
        tree, info = honest_tree(X, y, W=W, kind=kind, min_leaf=min_leaf,
                                 alpha=alpha, pi=pi, max_depth=max_depth,
                                 seed=seed * 7919 + b, subsample=sub)
        trees.append(tree)
        bag = [False] * n
        for i in sub:
            bag[i] = True
        bags.append(bag)
    return trees, bags, s


def forest_weights(trees, X, x):
    r"""Eq. (3): the forest's adaptive neighbourhood of x.

    .. math:: \alpha_{bi}(x) = \frac{1\{X_i \in L_b(x)\}}{|L_b(x)|},
              \qquad \alpha_i(x) = \frac1B \sum_b \alpha_{bi}(x).

    The leaf population is the I sample -- the rows the tree was allowed
    to average -- so the weights inherit honesty from the tree. They sum
    to 1 by construction, which is the cheapest way to catch a leaf
    bookkeeping error.
    """
    n = len(X)
    w = [0.0] * n
    B = len(trees)
    if B == 0:
        raise ValueError("hntfst: no trees")
    used = 0
    for tree in trees:
        node, _ = leaf_of(tree, x)
        rows = node["I"]
        if not rows:
            continue
        used += 1
        share = 1.0 / len(rows)
        for i in rows:
            w[i] += share
    if used == 0:
        raise ValueError("hntfst: every leaf is empty")
    return [v / used for v in w]


def cheatsheet():
    return ("hntfst: honest forest. Procedure 1 splits the subsample "
            "into I and J, places splits with J's responses and I's "
            "features but NEVER I's responses, and estimates leaves "
            "from I alone (Def. 2). Procedure 2 splits on W instead of "
            "Y. Def. 3: each feature has prob >= pi/d of being split "
            "on. Variance is the IJ, eq. (8), with the "
            "n(n-1)/(n-s)^2 correction for subsampling without "
            "replacement.")


# compact alias per ledger/NAMING.md
honestforest = honest_forest

# public names resolved by fn/_lazy_map.json
honest_random_forest = honest_forest
