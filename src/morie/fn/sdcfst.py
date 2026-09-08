"""Cross-fitted doubly robust treatment effects with an honest forest.

Two ideas do the work here, and they are separable, so both are
implemented and can be switched independently.

CROSS-FITTING. Estimating a nuisance function and then plugging it into
a score computed on the SAME observations makes the score's error
correlated with the nuisance error, and the bias that follows does not
vanish at root-n. Chernozhukov et al.'s answer is to split the sample:
fit the nuisance on the other folds, evaluate the score on this one, and
average. Nothing about the estimator changes -- only which data saw the
nuisance fit. The module reports the per-fold estimates as well as the
average, because a fold that disagrees with the rest is information
about the nuisance model, and averaging it away hides that.

NEYMAN ORTHOGONALITY. The score has to be insensitive to small errors in
the nuisance functions -- its derivative with respect to them must be
zero at the truth. That is what makes cross-fitting enough. Four scores
are offered so the difference is visible rather than asserted:

  "aipw"             the doubly robust score: outcome regressions plus
                     an inverse-propensity correction on the residuals.
                     Orthogonal, and consistent if EITHER the outcome
                     model or the propensity model is right.
  "partialling_out"  Robinson's partially linear score: residualise both
                     y and D on X and regress one on the other. Also
                     orthogonal, but it estimates a partially linear
                     coefficient, which equals the ATE only when the
                     effect is constant.
  "ipw"              inverse-propensity weighting alone. NOT orthogonal;
                     included as the baseline whose sensitivity to the
                     propensity model the other two are meant to remove.
  "plugin"           the outcome regressions alone, no correction. Also
                     not orthogonal, and the pair with "ipw" shows the
                     two ways to be wrong that "aipw" repairs.

The nuisance learner is separate again:

  "forest"  an HONEST regression forest, which is where the "semi" in
            the module's name comes from. Each tree splits on one
            subsample and fills its leaves from a DISJOINT one, so a
            leaf's value is not fitted to the points that chose the
            split. Without that, a forest's in-leaf averages are biased
            towards their own training points and the cross-fitted
            score inherits the bias it was meant to remove.
  "linear"  least squares for the outcomes and ridge-penalised logistic
            regression for the propensity. Fast, and the right choice
            when the confounding really is linear; also the arm that
            makes the forest's contribution measurable.

Propensities are trimmed away from 0 and 1 and the number of trimmed
units is REPORTED. A propensity of 0.001 turns one observation into a
thousand, and an estimator that silently does that is not robust, it is
lucky.

References
  Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W. and Robins, J. (2018) "Double/debiased machine learning
    for treatment and structural parameters." The Econometrics Journal
    21(1), C1-C68. Cross-fitting, Neyman orthogonality, and both the
    AIPW and partialling-out scores.
  Robinson, P.M. (1988) "Root-N-consistent semiparametric regression."
    Econometrica 56(4), 931-954. The partially linear score.
  Robins, J.M., Rotnitzky, A. and Zhao, L.P. (1994) "Estimation of
    regression coefficients when some regressors are not always
    observed." JASA 89(427), 846-866. The doubly robust score.
  Wager, S. and Athey, S. (2018) "Estimation and inference of
    heterogeneous treatment effects using random forests." JASA
    113(523), 1228-1242. Honest splitting.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["semi_doubly_robust_forest", "sdcfst", "honest_forest",
           "forest_predict", "logistic_fit", "SCORES", "LEARNERS",
           "cheatsheet"]

SCORES = ("aipw", "partialling_out", "ipw", "plugin")
LEARNERS = ("forest", "linear")


def _best_split(X, y, rows, feats, min_leaf):
    """Variance-reduction split over the given rows and features.

    Candidate cuts are midpoints between consecutive DISTINCT values of
    a feature among the rows in the node, which is the only set of cuts
    that can produce different partitions -- trying every value would
    repeat work and, worse, put a cut exactly on a data point, where the
    two arms' comparisons could part company.
    """
    best = None
    n = len(rows)
    if n < 2 * min_leaf:
        return None
    for f in feats:
        vals = sorted(set(X[i][f] for i in rows))
        if len(vals) < 2:
            continue
        order = sorted(rows, key=lambda i: (X[i][f], i))
        ys = [y[i] for i in order]
        # Running sums from the left; the right side is the complement,
        # so one pass gives every candidate's SSE.
        tot = _w.csum(ys)
        tot2 = _w.csum(v * v for v in ys)
        sl = 0.0
        sl2 = 0.0
        for k in range(n - 1):
            sl += ys[k]
            sl2 += ys[k] * ys[k]
            nl = k + 1
            nr = n - nl
            if nl < min_leaf or nr < min_leaf:
                continue
            if X[order[k]][f] == X[order[k + 1]][f]:
                continue
            sr = tot - sl
            sr2 = tot2 - sl2
            sse = (sl2 - sl * sl / nl) + (sr2 - sr * sr / nr)
            cut = 0.5 * (X[order[k]][f] + X[order[k + 1]][f])
            if best is None or sse < best[0] - 1e-15:
                best = (sse, f, cut)
    return best


def _grow(X, y, struct_rows, leaf_rows, feats_n, min_leaf, max_depth,
          rng, depth=0):
    """One honest tree: split on struct_rows, fill leaves from leaf_rows."""
    p = len(X[0])
    if (depth >= max_depth or len(struct_rows) < 2 * min_leaf
            or not leaf_rows):
        val = (_w.csum(y[i] for i in leaf_rows) / len(leaf_rows)
               if leaf_rows else
               _w.csum(y[i] for i in struct_rows) / len(struct_rows))
        return {"leaf": True, "value": val, "n": len(leaf_rows)}
    # mtry features, drawn WITHOUT replacement so a small mtry cannot
    # waste its draws on the same column twice.
    pool = list(range(p))
    feats = []
    for _ in range(min(feats_n, p)):
        j = int(rng.uniform() * len(pool))
        if j >= len(pool):
            j = len(pool) - 1
        feats.append(pool.pop(j))
    feats.sort()
    sp = _best_split(X, y, struct_rows, feats, min_leaf)
    if sp is None:
        val = _w.csum(y[i] for i in leaf_rows) / len(leaf_rows)
        return {"leaf": True, "value": val, "n": len(leaf_rows)}
    _, f, cut = sp
    sl = [i for i in struct_rows if X[i][f] <= cut]
    sr = [i for i in struct_rows if X[i][f] > cut]
    ll = [i for i in leaf_rows if X[i][f] <= cut]
    lr = [i for i in leaf_rows if X[i][f] > cut]
    if not ll or not lr:
        val = _w.csum(y[i] for i in leaf_rows) / len(leaf_rows)
        return {"leaf": True, "value": val, "n": len(leaf_rows)}
    return {"leaf": False, "feature": f, "cut": cut,
            "left": _grow(X, y, sl, ll, feats_n, min_leaf, max_depth,
                          rng, depth + 1),
            "right": _grow(X, y, sr, lr, feats_n, min_leaf, max_depth,
                           rng, depth + 1)}


def _tree_predict(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["cut"] \
            else node["right"]
    return node["value"]


def honest_forest(X, y, rows, n_trees=20, mtry=None, min_leaf=5,
                  max_depth=6, seed=1, rng=None):
    """Grow an honest regression forest on the given rows.

    Each tree draws a subsample and splits it in HALF: one half chooses
    the splits, the other fills the leaves. That is the honesty
    condition -- a leaf's value never sees the points that put it there.
    """
    if rng is None:
        rng = _core._SplitMix64(seed)
    p = len(X[0])
    if mtry is None:
        mtry = max(1, int(math.sqrt(p) + 0.5))
    trees = []
    m = len(rows)
    half = m // 2
    for _ in range(int(n_trees)):
        # Subsample without replacement by a partial Fisher-Yates over a
        # copy, which consumes exactly m - 1 uniforms whatever the data.
        pool = list(rows)
        for k in range(len(pool) - 1, 0, -1):
            j = int(rng.uniform() * (k + 1))
            if j > k:
                j = k
            pool[k], pool[j] = pool[j], pool[k]
        struct_rows = sorted(pool[:half])
        leaf_rows = sorted(pool[half:])
        trees.append(_grow(X, y, struct_rows, leaf_rows, mtry, min_leaf,
                           max_depth, rng))
    return {"trees": trees, "mtry": mtry, "min_leaf": min_leaf,
            "max_depth": max_depth}


def forest_predict(forest, x):
    """Average of the trees' leaf values at x."""
    return (_w.csum(_tree_predict(t, x) for t in forest["trees"])
            / len(forest["trees"]))


def logistic_fit(X, z, rows, ridge=1e-6, iters=50):
    """Ridge-penalised logistic regression by Newton-Raphson.

    The ridge term is not a modelling flourish: with a small fold and a
    separating covariate the unpenalised likelihood has no maximum, the
    Hessian goes singular and the two arms would fail in different
    places. A fixed tiny ridge makes the problem well posed in both.
    """
    p = len(X[0]) + 1
    beta = [0.0] * p

    def design(i):
        return [1.0] + [float(v) for v in X[i]]

    for _ in range(int(iters)):
        g = [0.0] * p
        h = [[0.0] * p for _ in range(p)]
        for i in rows:
            d = design(i)
            eta = _w.dot(d, beta)
            if eta > 30.0:
                eta = 30.0
            elif eta < -30.0:
                eta = -30.0
            mu = 1.0 / (1.0 + math.exp(-eta))
            wgt = mu * (1.0 - mu)
            if wgt < 1e-10:
                wgt = 1e-10
            r = z[i] - mu
            for a in range(p):
                g[a] += d[a] * r
                for b in range(p):
                    h[a][b] += wgt * d[a] * d[b]
        for a in range(p):
            g[a] -= ridge * beta[a]
            h[a][a] += ridge
        step = _w.solve_chol(_w.chol(h), g)
        beta = [beta[a] + step[a] for a in range(p)]
        if max(abs(v) for v in step) < 1e-12:
            break
    return beta


def _logit_predict(beta, x):
    eta = beta[0] + _w.dot(beta[1:], [float(v) for v in x])
    if eta > 30.0:
        eta = 30.0
    elif eta < -30.0:
        eta = -30.0
    return 1.0 / (1.0 + math.exp(-eta))


def _folds(n, k, rng):
    """Fold labels from a shuffle, so folds are balanced by construction."""
    idx = list(range(n))
    for t in range(len(idx) - 1, 0, -1):
        j = int(rng.uniform() * (t + 1))
        if j > t:
            j = t
        idx[t], idx[j] = idx[j], idx[t]
    lab = [0] * n
    for pos, i in enumerate(idx):
        lab[i] = pos % k
    return lab


def semi_doubly_robust_forest(y, D, X, K_fold=5, score="aipw",
                              learner="forest", n_trees=20, mtry=None,
                              min_leaf=5, max_depth=6, trim=0.02,
                              seed=1, ridge=1e-6):
    """Cross-fitted treatment effect with honest-forest nuisances.

    Parameters
    ----------
    y : sequence
        Outcome.
    D : sequence
        Binary treatment, 0 or 1.
    X : sequence of sequences
        Covariates.
    K_fold : int
        Number of cross-fitting folds. K = 1 means no cross-fitting at
        all, which is offered so the bias it removes can be measured.
    score : str
        "aipw", "partialling_out", "ipw" or "plugin".
    learner : str
        "forest" or "linear".
    n_trees, mtry, min_leaf, max_depth :
        Forest settings.
    trim : float
        Propensities are clipped into [trim, 1 - trim] and the number
        clipped is reported.
    seed : int
        Seed for the generator shared with the R arm.
    ridge : float
        Ridge penalty for the logistic propensity.

    Returns
    -------
    RichResult
        The estimate, its standard error from the influence function,
        the per-fold estimates, the trimming count and the fitted
        propensity summary.

    References
    ----------
    Chernozhukov et al. (2018) Econometrics Journal 21(1), C1-C68;
    Robinson (1988) Econometrica 56(4), 931-954; Robins, Rotnitzky and
    Zhao (1994) JASA 89(427), 846-866; Wager and Athey (2018) JASA
    113(523), 1228-1242.
    """
    if score not in SCORES:
        raise ValueError("score must be one of %r" % (SCORES,))
    if learner not in LEARNERS:
        raise ValueError("learner must be one of %r" % (LEARNERS,))
    yv = [float(v) for v in y]
    dv = [float(v) for v in D]
    Xv = [[float(v) for v in row] for row in X]
    n = len(yv)
    if len(dv) != n or len(Xv) != n:
        raise ValueError("y, D and X must have the same length")
    if any(v not in (0.0, 1.0) for v in dv):
        raise ValueError("D must be binary")
    if n < 8:
        raise ValueError("need at least eight observations")
    K = int(K_fold)
    if K < 1 or K > n:
        raise ValueError("K_fold must lie in 1..n")

    rng = _core._SplitMix64(seed)
    lab = [0] * n if K == 1 else _folds(n, K, rng)

    ps = [0.0] * n
    m0 = [0.0] * n
    m1 = [0.0] * n
    mall = [0.0] * n

    for k in range(K):
        te = [i for i in range(n) if lab[i] == k]
        tr = [i for i in range(n) if lab[i] != k] if K > 1 else te
        tr1 = [i for i in tr if dv[i] == 1.0]
        tr0 = [i for i in tr if dv[i] == 0.0]
        if not tr1 or not tr0:
            raise ValueError("a fold left one treatment arm empty; use "
                             "fewer folds")
        if learner == "forest":
            f1 = honest_forest(Xv, yv, tr1, n_trees, mtry, min_leaf,
                               max_depth, rng=rng)
            f0 = honest_forest(Xv, yv, tr0, n_trees, mtry, min_leaf,
                               max_depth, rng=rng)
            fa = honest_forest(Xv, yv, tr, n_trees, mtry, min_leaf,
                               max_depth, rng=rng)
            fd = honest_forest(Xv, dv, tr, n_trees, mtry, min_leaf,
                               max_depth, rng=rng)
            for i in te:
                m1[i] = forest_predict(f1, Xv[i])
                m0[i] = forest_predict(f0, Xv[i])
                mall[i] = forest_predict(fa, Xv[i])
                ps[i] = forest_predict(fd, Xv[i])
        else:
            des = [[1.0] + Xv[i] for i in range(n)]
            b1 = _w.ols([yv[i] for i in tr1], [des[i] for i in tr1])
            b0 = _w.ols([yv[i] for i in tr0], [des[i] for i in tr0])
            ba = _w.ols([yv[i] for i in tr], [des[i] for i in tr])
            bp = logistic_fit(Xv, dv, tr, ridge)
            for i in te:
                m1[i] = _w.dot(des[i], b1["beta"])
                m0[i] = _w.dot(des[i], b0["beta"])
                mall[i] = _w.dot(des[i], ba["beta"])
                ps[i] = _logit_predict(bp, Xv[i])

    trimmed = 0
    for i in range(n):
        if ps[i] < trim:
            ps[i] = trim
            trimmed += 1
        elif ps[i] > 1.0 - trim:
            ps[i] = 1.0 - trim
            trimmed += 1

    if score == "partialling_out":
        # Robinson: residualise both sides on X, then regress. The
        # coefficient is a ratio of averages, and its influence function
        # is the residual score divided by the mean squared treatment
        # residual.
        vres = [dv[i] - ps[i] for i in range(n)]
        ures = [yv[i] - mall[i] for i in range(n)]
        num = _w.csum(vres[i] * ures[i] for i in range(n))
        den = _w.csum(vres[i] * vres[i] for i in range(n))
        if den <= 0.0:
            raise ValueError("no variation left in the treatment after "
                             "residualising")
        est = num / den
        psi = [vres[i] * (ures[i] - est * vres[i]) / (den / n)
               for i in range(n)]
    else:
        psi = []
        for i in range(n):
            if score == "aipw":
                v = (m1[i] - m0[i]
                     + dv[i] * (yv[i] - m1[i]) / ps[i]
                     - (1.0 - dv[i]) * (yv[i] - m0[i]) / (1.0 - ps[i]))
            elif score == "ipw":
                v = (dv[i] * yv[i] / ps[i]
                     - (1.0 - dv[i]) * yv[i] / (1.0 - ps[i]))
            else:
                v = m1[i] - m0[i]
            psi.append(v)
        est = _w.csum(psi) / n

    var = _w.csum((v - est) * (v - est) for v in psi) / (n * (n - 1)) \
        if n > 1 else float("nan")
    se = math.sqrt(var) if var == var and var >= 0.0 else float("nan")

    fold_est = []
    for k in range(K):
        te = [i for i in range(n) if lab[i] == k]
        if te:
            fold_est.append(_w.csum(psi[i] for i in te) / len(te))

    z = est / se if se > 0.0 else float("nan")
    return RichResult(payload={
        "estimate": est,
        "se": se,
        "z": z,
        "p": (2.0 * (1.0 - _w.ncdf(abs(z))) if z == z else float("nan")),
        "ci_lower": est - 1.959963984540054 * se,
        "ci_upper": est + 1.959963984540054 * se,
        "fold_estimates": fold_est,
        "influence": psi,
        "propensity": ps,
        "m1": m1,
        "m0": m0,
        "trimmed": trimmed,
        "min_propensity": min(ps),
        "max_propensity": max(ps),
        "n": n,
        "n_treated": int(_w.csum(dv)),
        "K_fold": K,
        "score": score,
        "learner": learner,
        "seed": int(seed),
        "method": "cross-fitted doubly robust treatment effect",
    })


sdcfst = semi_doubly_robust_forest


def cheatsheet():
    return ("sdcfst: cross-fitted doubly robust treatment effects. "
            "scores " + ", ".join(SCORES) + "; learners "
            + ", ".join(LEARNERS))
