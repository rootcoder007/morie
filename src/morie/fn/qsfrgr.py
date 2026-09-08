"""Quantile survival forest: conditional survival curves and quantiles.

A random forest for survival does not average predictions. It averages
NEIGHBOURHOODS. Each tree puts the query point in a leaf, and the
training observations sharing that leaf are the ones the tree considers
comparable to it; averaging that membership over the forest gives a
weight for every training observation,

    alpha_i(x) = (1/B) sum_b 1{i in leaf_b(x)} / |leaf_b(x)|

and those weights, which sum to one, define a conditional distribution.
Feed them to a Kaplan-Meier estimator and you get a conditional survival
curve; invert the curve and you get a conditional quantile -- the median
survival time for a patient like this one. That is the whole method, and
it is why a forest can estimate a quantile at all: a quantile is not an
average of anything, so a forest that averaged predictions could not
produce one.

Censoring is what makes survival different and it is handled where it
belongs, inside the Kaplan-Meier product rather than by dropping the
censored rows. An observation censored at time t contributes to every
risk set up to t and to no death. Dropping it instead would bias the
curve upward, badly, and is the single most common way this goes wrong.

Splitting uses the LOG-RANK statistic between the two children, which is
the standard survival criterion: it asks whether the two groups' hazards
differ, rather than whether their mean times do, and it is therefore
undisturbed by censoring in a way a variance-reduction split is not.

Honesty is a route, not a default assumption. An honest tree uses one
half of the sample to choose the splits and the OTHER half to populate
the leaves, so the values in a leaf were not used to decide that the
leaf should exist. Without it the leaf estimates are biased towards
whatever the splitting found, which is exactly the overfitting the
confidence statements would then understate. It costs half the data and
it is worth it, so it is on by default and can be turned off to see the
difference.

References
  Cui, Y., Kosorok, M.R., Sverdrup, E., Wager, S. and Zhu, R. (2023)
    "Estimating heterogeneous treatment effects with right-censored data
    via causal survival forests." Journal of the Royal Statistical
    Society Series B 85(2), 179-211. doi:10.1093/jrsssb/qkac001. The
    forest-weighted survival machinery.
  Wager, S. and Athey, S. (2018) "Estimation and inference of
    heterogeneous treatment effects using random forests." Journal of
    the American Statistical Association 113(523), 1228-1242. Honesty
    and the forest-as-weights view.
  Athey, S., Tibshirani, J. and Wager, S. (2019) "Generalized random
    forests." The Annals of Statistics 47(2), 1148-1178. The weighting
    formula above.
  Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S. (2008)
    "Random survival forests." The Annals of Applied Statistics 2(3),
    841-860. The log-rank splitting rule.
  Kaplan, E.L. and Meier, P. (1958) "Nonparametric estimation from
    incomplete observations." Journal of the American Statistical
    Association 53(282), 457-481.
  Meinshausen, N. (2006) "Quantile regression forests." Journal of
    Machine Learning Research 7, 983-999. Forests for quantiles rather
    than means.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["qsfrgr", "quantile_survival_forest", "survival_forest",
           "forest_weights", "weighted_km", "km_quantile", "logrank",
           "SPLITS", "cheatsheet"]

SPLITS = ("logrank", "events")


def logrank(time, event, left, right):
    """The log-rank statistic between two groups of row indices.

    Returns the squared standardised difference between observed and
    expected deaths in the left group, which is the quantity a survival
    tree maximises. Zero when the groups are indistinguishable and
    undefined -- reported as zero -- when neither group can contribute
    a comparison.
    """
    rows = sorted(list(left) + list(right))
    if not left or not right:
        return 0.0
    times = sorted(set(time[i] for i in rows if event[i]))
    o_minus_e = []
    var = []
    for t in times:
        n1 = 0
        n2 = 0
        d1 = 0
        d2 = 0
        for i in left:
            if time[i] >= t:
                n1 += 1
                if event[i] and time[i] == t:
                    d1 += 1
        for i in right:
            if time[i] >= t:
                n2 += 1
                if event[i] and time[i] == t:
                    d2 += 1
        n = n1 + n2
        d = d1 + d2
        if n < 2 or d == 0:
            continue
        e1 = d * n1 / float(n)
        o_minus_e.append(d1 - e1)
        var.append(d * (n1 / float(n)) * (n2 / float(n))
                   * (n - d) / float(n - 1))
    if not var:
        return 0.0
    v = _w.csum(var)
    if v <= 0.0:
        return 0.0
    s = _w.csum(o_minus_e)
    return s * s / v


def _events_in(event, rows):
    n = 0
    for i in rows:
        if event[i]:
            n += 1
    return n


def _best_split(X, time, event, rows, feats, min_leaf, rule):
    """The (feature, threshold) maximising the criterion, or None."""
    best = None
    for f in feats:
        vals = sorted(set(X[i][f] for i in rows))
        if len(vals) < 2:
            continue
        for k in range(len(vals) - 1):
            thr = 0.5 * (vals[k] + vals[k + 1])
            left = [i for i in rows if X[i][f] <= thr]
            right = [i for i in rows if X[i][f] > thr]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            if rule == "logrank":
                score = logrank(time, event, left, right)
            else:
                # A split that isolates events is better than one that
                # isolates censoring; this is the cheap rule and it is
                # here to be visibly worse than the log-rank one.
                a = _events_in(event, left)
                b = _events_in(event, right)
                score = abs(a / float(len(left)) - b / float(len(right)))
            if best is None or score > best[2] or (
                    score == best[2] and (f, thr) < (best[0], best[1])):
                best = (f, thr, score)
    return best


def _grow(X, time, event, struct_rows, leaf_rows, feats_n, min_leaf,
          max_depth, depth, rng, rule):
    node = {"leaf": True, "rows": list(leaf_rows)}
    if (depth >= max_depth or len(struct_rows) < 2 * min_leaf
            or _events_in(event, struct_rows) < 2):
        return node
    p = len(X[0])
    feats = []
    pool = list(range(p))
    for _ in range(min(feats_n, p)):
        # int(u * len) clamped, not an integer draw: it is the idiom the
        # rest of the forest code in this package already uses, and two
        # different ways of turning a uniform into an index consume the
        # stream identically but land on different features.
        j = int(rng.uniform() * len(pool))
        if j >= len(pool):
            j = len(pool) - 1
        feats.append(pool.pop(j))
    feats.sort()
    sp = _best_split(X, time, event, struct_rows, feats, min_leaf, rule)
    if sp is None or sp[2] <= 0.0:
        return node
    f, thr, _ = sp
    sl = [i for i in struct_rows if X[i][f] <= thr]
    sr = [i for i in struct_rows if X[i][f] > thr]
    ll = [i for i in leaf_rows if X[i][f] <= thr]
    lr = [i for i in leaf_rows if X[i][f] > thr]
    if not ll or not lr:
        return node
    return {"leaf": False, "f": f, "thr": thr,
            "l": _grow(X, time, event, sl, ll, feats_n, min_leaf,
                       max_depth, depth + 1, rng, rule),
            "r": _grow(X, time, event, sr, lr, feats_n, min_leaf,
                       max_depth, depth + 1, rng, rule)}


def _leaf_of(node, x):
    while not node["leaf"]:
        node = node["l"] if x[node["f"]] <= node["thr"] else node["r"]
    return node["rows"]


def survival_forest(X, time, event, n_trees=20, mtry=None, min_leaf=3,
                    max_depth=6, honest=True, seed=0, rule="logrank"):
    """Grow a survival forest and return its trees.

    Each tree draws a subsample, splits it in half when honest, grows on
    the first half and fills the leaves from the second.
    """
    if rule not in SPLITS:
        raise ValueError("rule must be one of %r" % (SPLITS,))
    n = len(time)
    p = len(X[0])
    m = p if mtry is None else int(mtry)
    rng = _core._SplitMix64(seed)
    trees = []
    for _ in range(int(n_trees)):
        # Subsample without replacement, drawn by a fixed shuffle so the
        # two arms take the same rows in the same order.
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            j = int(rng.uniform() * (i + 1))
            if j > i:
                j = i
            idx[i], idx[j] = idx[j], idx[i]
        take = idx[:max(2 * min_leaf, n // 2)]
        if honest:
            h = len(take) // 2
            struct = sorted(take[:h])
            leaf = sorted(take[h:])
        else:
            struct = sorted(take)
            leaf = sorted(take)
        if not struct or not leaf:
            continue
        trees.append(_grow(X, time, event, struct, leaf, m, min_leaf,
                           int(max_depth), 0, rng, rule))
    return trees


def forest_weights(trees, x, n):
    """The forest's weight on each training observation, summing to one.

    Each tree contributes one over its leaf size to every observation in
    that leaf; the forest averages over trees. A tree whose leaf is
    empty contributes nothing rather than dividing by zero.
    """
    w = [0.0] * n
    used = 0
    for t in trees:
        rows = _leaf_of(t, x)
        if not rows:
            continue
        used += 1
        c = 1.0 / len(rows)
        for i in rows:
            w[i] += c
    if used == 0:
        return w, 0
    for i in range(n):
        w[i] = w[i] / used
    return w, used


def weighted_km(time, event, weights, grid=None):
    """Weighted Kaplan-Meier survival curve.

    The product over event times of one minus the weighted deaths over
    the weighted risk set. A censored observation stays in the risk set
    until its censoring time and never enters a numerator, which is the
    entire content of "handling censoring".
    """
    n = len(time)
    ts = sorted(set(time[i] for i in range(n)
                    if event[i] and weights[i] > 0.0))
    s = 1.0
    curve = []
    for t in ts:
        d = _w.csum(weights[i] for i in range(n)
                    if event[i] and time[i] == t)
        r = _w.csum(weights[i] for i in range(n) if time[i] >= t)
        if r <= 0.0:
            continue
        s = s * (1.0 - d / r)
        curve.append((t, s, d, r))
    if grid is None:
        return curve
    out = []
    for g in grid:
        v = 1.0
        for t, sv, _, _ in curve:
            if t <= g:
                v = sv
        out.append(v)
    return curve, out


def km_quantile(curve, q):
    """The smallest time whose survival has fallen to or below 1 - q.

    Returns None when the curve never gets there, which is the honest
    answer for a median that is not reached rather than the largest
    observed time dressed up as an estimate.
    """
    if not 0.0 < float(q) < 1.0:
        raise ValueError("the quantile must lie strictly inside (0, 1)")
    target = 1.0 - float(q)
    for t, s, _, _ in curve:
        if s <= target:
            return t
    return None


def quantile_survival_forest(time, event, X, quantile=0.5, n_trees=20,
                             mtry=None, min_leaf=3, max_depth=6,
                             honest=True, seed=0, rule="logrank",
                             newX=None, grid=None):
    """Conditional survival quantiles from a forest.

    Parameters
    ----------
    time, event : sequence
        Observed time and the event indicator, one is a death and zero
        a censoring.
    X : sequence of sequences
        Covariates, one row per observation.
    quantile : float
        Which conditional quantile of the survival distribution.
    newX : sequence of sequences or None
        Points to predict at. The training rows themselves when
        omitted.
    grid : sequence or None
        Times at which the conditional survival curve is reported.

    Returns
    -------
    RichResult
        The conditional quantile at each query point, the survival
        curve on the grid, the forest weights' effective sample size,
        and how many observations were censored.

    References
    ----------
    Cui et al. (2023) JRSS-B 85(2), 179-211; Athey, Tibshirani and
    Wager (2019) Ann Statist 47(2), 1148-1178.
    """
    t = [float(v) for v in time]
    e = [1 if v else 0 for v in event]
    xs = [[float(v) for v in row] for row in X]
    n = len(t)
    if n < 4:
        raise ValueError("need at least four observations")
    if len(e) != n or len(xs) != n:
        raise ValueError("time, event and X must agree in length")
    trees = survival_forest(xs, t, e, n_trees, mtry, min_leaf, max_depth,
                            honest, seed, rule)
    if not trees:
        raise ValueError("no tree could be grown; the sample is too "
                         "small for the leaf size")
    qx = xs if newX is None else [[float(v) for v in r] for r in newX]
    if grid is None:
        grid = sorted(set(t))
    grid = [float(g) for g in grid]

    quants = []
    curves = []
    ess = []
    for x in qx:
        w, used = forest_weights(trees, x, n)
        curve, sg = weighted_km(t, e, w, grid)
        quants.append(km_quantile(curve, quantile))
        curves.append(sg)
        # Effective sample size of the weights: one over the sum of
        # their squares. It says how many observations the estimate is
        # really resting on, which a weight vector of length n hides.
        ss = _w.csum(v * v for v in w)
        ess.append(1.0 / ss if ss > 0.0 else 0.0)

    got = [v for v in quants if v is not None]
    return RichResult(payload={
        "quantile_estimate": [float("nan") if v is None else v
                              for v in quants],
        "n_unreached": len(quants) - len(got),
        "curve": curves,
        "grid": grid,
        "ess": ess,
        "mean_ess": _w.csum(ess) / len(ess) if ess else float("nan"),
        "estimate": _w.csum(got) / len(got) if got else float("nan"),
        "se": float("nan"),
        "n_trees": len(trees),
        "n": n,
        "n_events": sum(e),
        "n_censored": n - sum(e),
        "n_query": len(qx),
        "quantile": float(quantile),
        "honest": bool(honest),
        "rule": rule,
        "method": "quantile survival forest",
    })


qsfrgr = quantile_survival_forest


def cheatsheet():
    return ("qsfrgr: quantile survival forest. splits " + ", ".join(SPLITS)
            + "; forest weights into a weighted Kaplan-Meier, inverted")
