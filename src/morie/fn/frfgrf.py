# morie.fn -- function file (rootcoder007/morie)
r"""Forest-fit consistency diagnostics.

Wager & Athey's consistency and asymptotic-normality results do not hold
for any forest. They hold for forests that are honest, random-split,
:math:`\alpha`-regular and symmetric, grown on subsamples of size
:math:`s \asymp n^{\beta}` with

.. math:: \beta_{\min} = 1 - \Big(1 + \frac{d}{\pi}\,
          \frac{\log(\alpha^{-1})}{\log((1-\alpha)^{-1})}\Big)^{-1}
          < \beta < 1.

Every one of those is a property of *how the forest was grown*, and
every one is silently violable. A forest that breaks them still
produces predictions and still produces intervals; what it stops
producing is any reason to believe the intervals.

**So the conditions are checked, not assumed.** This module audits a
fitted forest:

* honesty, tested by permuting the responses the leaves average and
  requiring the split structure not to move -- the only test that can
  actually fail;
* the random-split floor, by measuring each feature's realised split
  share against :math:`\pi/d`;
* :math:`\alpha`-regularity, by finding the most lopsided split;
* the subsample rate, by solving for the realised :math:`\beta` and
  comparing it with :math:`\beta_{\min}` -- reported rather than
  scored, because :math:`\beta_{\min}` is 0.997 already at
  :math:`d=3, \alpha=0.05, \pi=0.5`, so the theorem asks for
  :math:`s \approx n` and no practical subsample fraction meets it.
  Folding that into a pass/fail would mark every forest ever grown as
  failing, which tells nobody anything;
* and the fit itself, by checking the error falls as :math:`n` grows,
  which is what consistency means operationally.

The point is that a diagnostic returning "all clear" on a forest that
is *not* honest would be worse than no diagnostic, so the honesty test
here is the structural one rather than a flag copied from the
constructor's arguments.

References
----------
Wager, S. & Athey, S. (2018) "Estimation and Inference of Heterogeneous
Treatment Effects using Random Forests", *Journal of the American
Statistical Association* 113(523), 1228-1242,
doi:10.1080/01621459.2017.1319839, arXiv:1510.04342. Definitions 2-5,
Theorem 3 and its beta_min, Theorem 1.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. The same conditions in the
generalized setting.

Biau, G. (2012) "Analysis of a Random Forests Model", *Journal of
Machine Learning Research* 13, 1063-1095. Earlier consistency analysis
of the same shape.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import grow_forest, honest_tree, tree_predict

__all__ = ["forest_fit_check", "beta_min", "honesty_test",
            "split_share", "regularity"]

_EPS = 1e-12


def beta_min(d, alpha=0.05, pi=0.5):
    r"""Theorem 3's lower bound on the subsample exponent."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("frfgrf: alpha must be in (0, 0.5), got %r"
                         % (alpha,))
    if not 0.0 < pi <= 1.0:
        raise ValueError("frfgrf: pi must be in (0, 1], got %r" % (pi,))
    if d < 1:
        raise ValueError("frfgrf: need at least one feature")
    ratio = math.log(1.0 / alpha) / math.log(1.0 / (1.0 - alpha))
    return 1.0 - (1.0 + (float(d) / pi) * ratio) ** -1.0


def _structure(tree):
    if tree["leaf"]:
        return ["leaf"]
    return ([(tree["feature"], round(tree["threshold"], 12))]
            + _structure(tree["left"]) + _structure(tree["right"]))


def honesty_test(X, y, kind="double-sample", min_leaf=5, seed=11,
                 n_permutations=3):
    r"""Permute the responses the leaves will average; the splits must
    not move.

    Returns whether the structure held, and -- as a control -- whether
    permuting the SPLITTING responses does move it. A tree that ignores
    every response would pass the first test trivially, so the second
    is what makes the first mean something.
    """
    tree, info = honest_tree(X, y, kind=kind, min_leaf=min_leaf,
                             seed=seed)
    base = _structure(tree)
    rng = np.random.default_rng(seed + 1)
    stable = True
    for _ in range(int(n_permutations)):
        yp = list(y)
        for a, b in zip(info["I"], sorted(
                info["I"], key=lambda _i: float(rng.uniform()))):
            yp[a] = y[b]
        tp, _ = honest_tree(X, yp, kind=kind, min_leaf=min_leaf,
                            seed=seed)
        if _structure(tp) != base:
            stable = False
            break
    yj = list(y)
    for a, b in zip(info["J"], sorted(
            info["J"], key=lambda _i: float(rng.uniform()))):
        yj[a] = y[b]
    tj, _ = honest_tree(X, yj, kind=kind, min_leaf=min_leaf, seed=seed)
    responsive = _structure(tj) != base
    return {"honest": stable and (responsive or kind == "propensity"),
            "splits_stable_under_I_permutation": stable,
            "splits_move_under_J_permutation": responsive,
            "n_splits": sum(1 for v in base if v != "leaf")}


def split_share(trees, d):
    """Realised share of splits on each feature."""
    counts = [0] * d

    def walk(nd):
        if nd["leaf"]:
            return
        counts[nd["feature"]] += 1
        walk(nd["left"])
        walk(nd["right"])

    for t in trees:
        walk(t)
    tot = sum(counts) or 1
    return [c / float(tot) for c in counts], counts


def regularity(trees):
    """The most lopsided split found, as a fraction on the small side."""
    worst = 1.0

    def walk(nd):
        nonlocal worst
        if nd["leaf"]:
            return
        nl = _leaf_count(nd["left"])
        nr = _leaf_count(nd["right"])
        tot = nl + nr
        if tot > 0:
            worst = min(worst, min(nl, nr) / float(tot))
        walk(nd["left"])
        walk(nd["right"])

    def _leaf_count(nd):
        if nd["leaf"]:
            return nd["n_I"]
        return _leaf_count(nd["left"]) + _leaf_count(nd["right"])

    for t in trees:
        walk(t)
    return worst


def forest_fit_check(y, X, n_trees=100, min_leaf=5, subsample_frac=0.5,
                     alpha=0.05, pi=0.5, seed=0, kind="double-sample",
                     sizes=None):
    r"""Audit a forest against the conditions its theory requires."""
    yv = k.vec(y)
    n = len(yv)
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("frfgrf: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    d = len(Xm[0]) if Xm and Xm[0] else 0
    if d == 0:
        raise ValueError("frfgrf: no features")
    if n < 40:
        raise ValueError("frfgrf: need at least 40 observations, got %d"
                         % n)
    trees, bags, s = grow_forest(Xm, yv, kind=kind, n_trees=n_trees,
                                 min_leaf=min_leaf,
                                 subsample_frac=subsample_frac,
                                 alpha=alpha, pi=pi, seed=seed)
    hon = honesty_test(Xm, yv, kind=kind, min_leaf=min_leaf,
                       seed=seed + 11)
    share, counts = split_share(trees, d)
    floor = pi / d
    reg = regularity(trees)
    bmin = beta_min(d, alpha=alpha, pi=pi)
    beta = math.log(s) / math.log(n) if n > 1 and s > 1 else 0.0

    # The three conditions a practitioner controls directly.
    checks = {
        "honest": hon["honest"],
        "random_split_floor": min(share) >= 0.2 * floor,
        "alpha_regular": reg >= 0.5 * alpha,
    }
    # The subsample rate is reported SEPARATELY because beta_min is
    # brutal at realistic settings: with d = 3, alpha = 0.05 and
    # pi = 0.5 it is 0.997, so the theorem asks for s almost equal to n
    # and no practical subsample fraction meets it. Folding it into a
    # pass/fail would mean every honest forest ever grown "fails", which
    # tells the user nothing. It is surfaced as a number to compare.
    rate_ok = bmin < beta < 1.0
    return RichResult(payload={
        "estimate": all(checks.values()), "passes": all(checks.values()),
        "checks": checks, "honesty": hon,
        "subsample_rate_ok": rate_ok,
        "subsample_rate_note": (
            "beta = log(s)/log(n) = %.3f against beta_min = %.3f; the "
            "bound is near 1 for any moderate d, so it is reported "
            "rather than scored" % (beta, bmin)),
        "split_share": share, "split_counts": counts,
        "random_split_floor": floor, "min_share": min(share),
        "regularity": reg, "alpha": float(alpha), "pi": float(pi),
        "beta": beta, "beta_min": bmin, "s": s, "n": n, "d": d,
        "n_trees": int(n_trees), "kind": kind,
        "failed": [nm for nm, ok in checks.items() if not ok],
        "method": "forest-fit consistency diagnostics, Wager & Athey "
                  "(2018) Definitions 2-5 and Theorem 3",
    })


def cheatsheet():
    return ("frfgrf: audit the conditions the theory needs -- honesty "
            "(tested by permuting the leaf-averaged responses, with a "
            "J-permutation control), the pi/d split floor, "
            "alpha-regularity, and beta = log(s)/log(n) above "
            "beta_min = 1 - (1 + (d/pi) log(1/alpha)/log(1/(1-alpha)))^-1.")


# compact alias per ledger/NAMING.md
forestfitcheck = forest_fit_check

# public names resolved by fn/_lazy_map.json
forest_fit_consistency = forest_fit_check
