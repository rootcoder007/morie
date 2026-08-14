# morie.fn -- function file (rootcoder007/morie)
r"""Variable importance for a CATE forest.

Which covariates does the forest actually use to distinguish treatment
effects? The natural answer is how often it splits on each one -- but a
raw split count is dominated by the deep splits, of which there are
exponentially many and which each govern a handful of observations.

**Depth weighting is what makes the count mean something.** A split at
depth :math:`k` is counted with weight proportional to
:math:`1/k^{\,\text{decay}}` and only the first ``max_depth`` levels are
counted at all:

.. math:: \mathrm{VI}_j \propto \sum_{k=1}^{K}
          \frac{1}{k^{\,d}}\;\frac{\#\{\text{depth-}k
          \text{ splits on } j\}}{\#\{\text{depth-}k\text{ splits}\}},

normalised to sum to 1. The inner ratio matters as much as the decay:
without it, level :math:`k` contributes :math:`2^k` splits and the
deepest counted level swamps everything above it whatever the weights.

**The forest must be grown on a RESIDUALISED outcome.** Form the
pseudo-outcome from raw Y and it still contains :math:`m(X)`, so the
splits chase the confounding surface and the ranking hands first place
to whichever covariate confounds hardest. Cross-fitted local centering
comes first, and the anchor's design puts the confounder and the effect
modifier in different columns precisely so that failure is visible.

**A split count alone cannot separate "used" from "useful".** Under
Definition 3 every feature is offered with probability at least
:math:`\pi/d`, so a covariate that predicts nothing still collects
splits -- and the anchor measures that floor rather than assuming it is
zero. The permutation route answers the different question of how much
the *predictions* depend on a covariate, and both are provided because
they disagree in an informative way: a covariate that is split on often
but only in leaves too small to move anything scores high on frequency
and low on permutation.

References
----------
Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. Eq. (3), and the split
frequency measure of its Sec. 6.

Wager, S. & Athey, S. (2018) "Estimation and Inference of Heterogeneous
Treatment Effects using Random Forests", *Journal of the American
Statistical Association* 113(523), 1228-1242,
doi:10.1080/01621459.2017.1319839. Definition 3, the pi/d floor that
puts a nonzero split share on every covariate.

Breiman, L. (2001) "Random Forests", *Machine Learning* 45(1), 5-32,
doi:10.1023/A:1010933404324. The permutation importance measure.

Strobl, C., Boulesteix, A.-L., Zeileis, A. & Hothorn, T. (2007) "Bias in
random forest variable importance measures: illustrations, sources and a
solution", *BMC Bioinformatics* 8, 25, doi:10.1186/1471-2105-8-25. Why a
raw split count is biased toward high-cardinality covariates.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest, tree_predict

__all__ = ["cate_variable_importance", "split_frequency_importance",
           "permutation_importance"]

_EPS = 1e-12


def _folds(n, V):
    V = max(2, min(int(V), n))
    return [[i for i in range(n) if i % V == v] for v in range(V)]


def _center(y, W, X, n_folds, n_trees, min_leaf, seed):
    """Cross-fitted m(X) and e(X), as in the partial-linear forest."""
    n = len(y)
    mh, eh = [0.0] * n, [0.0] * n
    for val in _folds(n, n_folds):
        tr = [i for i in range(n) if i not in set(val)]
        if not tr:
            continue
        Xt = [X[i] for i in tr]
        for src, dest in ((y, mh), (W, eh)):
            vt = [src[i] for i in tr]
            trees, _, _ = grow_forest(Xt, vt, n_trees=n_trees,
                                      min_leaf=min_leaf,
                                      seed=seed + (0 if dest is mh
                                                   else 1))
            for i in val:
                w = forest_weights(trees, Xt, X[i])
                dest[i] = sum(w[t] * vt[t] for t in range(len(tr)))
    return mh, eh


def _depth_counts(tree, max_depth, d):
    """Split counts by (depth, feature), depth 1 at the root."""
    counts = [[0.0] * d for _ in range(max_depth + 1)]

    def walk(nd, depth):
        if nd["leaf"] or depth > max_depth:
            return
        counts[depth][nd["feature"]] += 1.0
        walk(nd["left"], depth + 1)
        walk(nd["right"], depth + 1)

    walk(tree, 1)
    return counts


def split_frequency_importance(trees, d, max_depth=4, decay=2.0):
    r"""Depth-weighted split frequency, normalised to sum to 1.

    The share is taken WITHIN each depth before weighting: level k holds
    up to 2^k splits, so without that the deepest counted level decides
    the answer no matter what the decay is.
    """
    if max_depth < 1:
        raise ValueError("crfsel: max_depth must be at least 1, got %d"
                         % max_depth)
    if decay < 0.0:
        raise ValueError("crfsel: decay must be non-negative, got %r"
                         % (decay,))
    total = [0.0] * d
    for depth in range(1, max_depth + 1):
        at_depth = [0.0] * d
        for tree in trees:
            c = _depth_counts(tree, max_depth, d)
            for j in range(d):
                at_depth[j] += c[depth][j]
        tot = sum(at_depth)
        if tot <= 0.0:
            continue
        w = 1.0 / (depth ** decay)
        for j in range(d):
            total[j] += w * at_depth[j] / tot
    s = sum(total)
    return [v / s for v in total] if s > 0 else [1.0 / d] * d


def permutation_importance(trees, X, y, features=None, seed=0,
                           n_repeats=3):
    """The rise in weighted prediction error when a covariate is
    shuffled: how much the PREDICTIONS depend on it, which is a
    different question from how often it is split on."""
    n = len(y)
    d = len(X[0]) if n and X[0] else 0
    feats = list(range(d)) if features is None else list(features)

    def err(Xa):
        tot = 0.0
        for i in range(n):
            w = forest_weights(trees, X, Xa[i])
            pred = sum(w[t] * y[t] for t in range(n))
            tot += (y[i] - pred) ** 2
        return tot / n

    base = err(X)
    rng = np.random.default_rng(seed)
    out = [0.0] * d
    for j in feats:
        acc = 0.0
        for _ in range(int(n_repeats)):
            order = sorted(range(n), key=lambda _i: float(rng.uniform()))
            Xp = [list(X[i]) for i in range(n)]
            for i in range(n):
                Xp[i][j] = X[order[i]][j]
            acc += err(Xp) - base
        out[j] = acc / n_repeats
    return out, base


def cate_variable_importance(y, W, X, n_trees=200, min_leaf=5,
                             max_depth=4, decay=2.0, seed=0,
                             names=None, permutation=False):
    r"""Rank covariates by how the CATE forest uses them.

    The forest is grown on the treatment-residualised outcome, so the
    splits it makes are the ones that separate treatment EFFECTS rather
    than outcome levels.
    """
    yv, Wv = k.vec(y), k.vec(W)
    n = len(yv)
    if len(Wv) != n:
        raise ValueError("crfsel: %d outcomes but %d treatments"
                         % (n, len(Wv)))
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("crfsel: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    d = len(Xm[0]) if Xm and Xm[0] else 0
    if d == 0:
        raise ValueError("crfsel: no covariates")
    nm = (list(names) if names is not None
          else ["X%d" % (j + 1) for j in range(d)])
    if len(nm) != d:
        raise ValueError("crfsel: %d names for %d covariates"
                         % (len(nm), d))
    if n < 60:
        raise ValueError("crfsel: need at least 60 observations, got %d"
                         % n)

    # Local centering FIRST. Without it the pseudo-outcome still
    # carries m(X), so the forest splits on the confounding surface and
    # ranks the confounder above the effect modifier -- which is exactly
    # what the anchor caught when this used a global mean instead.
    mh, eh = _center(yv, Wv, Xm, n_folds=5,
                     n_trees=max(50, n_trees // 2), min_leaf=min_leaf,
                     seed=seed)
    yr = [yv[i] - mh[i] for i in range(n)]
    wr = [Wv[i] - eh[i] for i in range(n)]
    denom = sum(wr[i] * wr[i] for i in range(n)) / n
    if denom < _EPS:
        raise ValueError("crfsel: the treatment does not vary")
    pseudo = [wr[i] * yr[i] / denom for i in range(n)]
    trees, _, _ = grow_forest(Xm, pseudo, n_trees=n_trees,
                              min_leaf=min_leaf, seed=seed)
    freq = split_frequency_importance(trees, d, max_depth=max_depth,
                                      decay=decay)
    perm, base = ((None, None) if not permutation
                  else permutation_importance(trees, Xm, pseudo,
                                              seed=seed))
    order = sorted(range(d), key=lambda j: -freq[j])
    ranking = [{"variable": nm[j], "index": j, "importance": freq[j],
                "rank": r + 1,
                "permutation": (perm[j] if perm else None)}
               for r, j in enumerate(order)]
    return RichResult(payload={
        "estimate": freq, "importance": freq,
        "importance_by_name": {nm[j]: freq[j] for j in range(d)},
        "ranking": ranking, "top": nm[order[0]],
        "permutation": perm, "baseline_error": base,
        "n": n, "d": d, "n_trees": int(n_trees),
        "max_depth": int(max_depth), "decay": float(decay),
        "method": "depth-weighted split-frequency variable importance "
                  "for a CATE forest, Athey, Tibshirani & Wager (2019)",
    })


def cheatsheet():
    return ("crfsel: grow the forest on the treatment-residualised "
            "pseudo-outcome so splits separate EFFECTS; score each "
            "covariate by depth-weighted split share, taking the share "
            "WITHIN each depth first (level k holds 2^k splits). Under "
            "Def. 3 every covariate gets a nonzero floor, so a nonzero "
            "score is not evidence of use.")


# compact alias per ledger/NAMING.md
catevariableimportance = cate_variable_importance

# public names resolved by fn/_lazy_map.json
causal_forest_selection = cate_variable_importance
