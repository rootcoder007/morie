# morie.fn -- function file (rootcoder007/morie)
r"""IPW-augmented forest: doubly robust average effects.

A causal forest gives :math:`\hat\tau(x)` pointwise. Averaging it gives
an estimate of the average effect that inherits every bias in the
outcome model, because nothing in the average corrects for how treatment
was assigned.

**The augmented score fixes that, and only needs one of two models to be
right.** For each observation,

.. math:: \hat\Gamma_i = \hat\mu_1(X_i) - \hat\mu_0(X_i)
          + \frac{W_i\,(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)}
          - \frac{(1-W_i)(Y_i - \hat\mu_0(X_i))}{1 - \hat e(X_i)},

whose mean estimates the ATE. If the outcome models are right the
weighted residuals have mean zero and the first term carries the
estimate; if the propensity is right the residual terms correct
whatever the outcome models got wrong. The anchor breaks each model in
turn and checks the estimate survives -- and breaks both, where it
should not.

**Overlap is the assumption that actually bites.** The score divides by
:math:`\hat e` and :math:`1 - \hat e`, so a propensity near either
boundary produces an enormous term that no amount of averaging tames.
Trimming is applied and the largest weight is reported, because a
doubly robust estimator with one observation carrying a weight of 500
is a single-observation estimator wearing a robustness argument.

References
----------
Robins, J. M., Rotnitzky, A. & Zhao, L. P. (1994) "Estimation of
Regression Coefficients When Some Regressors Are Not Always Observed",
*Journal of the American Statistical Association* 89(427), 846-866,
doi:10.1080/01621459.1994.10476818. The augmented IPW score.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. The forest supplying the
outcome and propensity surfaces.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. & Robins, J. (2018) "Double/debiased machine learning for
treatment and structural parameters", *The Econometrics Journal* 21(1),
C1-C68, doi:10.1111/ectj.12097. Cross-fitting, and why it is needed
when the nuisances are machine-learned.

Crump, R. K., Hotz, V. J., Imbens, G. W. & Mitnik, O. A. (2009)
"Dealing with limited overlap in estimation of average treatment
effects", *Biometrika* 96(1), 187-199, doi:10.1093/biomet/asn055. The
overlap problem the trimming responds to.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest

__all__ = ["ipw_forest", "aipw_scores", "forest_nuisances"]

_EPS = 1e-12


def _folds(n, V):
    V = max(2, min(int(V), n))
    return [[i for i in range(n) if i % V == v] for v in range(V)]


def forest_nuisances(y, W, X, n_folds=5, n_trees=120, min_leaf=5,
                     seed=0):
    r"""Cross-fitted :math:`\hat\mu_1, \hat\mu_0, \hat e`."""
    n = len(y)
    mu1, mu0, e = [0.0] * n, [0.0] * n, [0.0] * n
    for val in _folds(n, n_folds):
        tr = [i for i in range(n) if i not in set(val)]
        if not tr:
            continue
        for arm, dest in ((1.0, mu1), (0.0, mu0)):
            idx = [i for i in tr if W[i] == arm]
            if len(idx) < 4 * min_leaf:
                raise ValueError("ipwgrf: too few training rows in "
                                 "treatment arm %g" % arm)
            Xa, ya = [X[i] for i in idx], [y[i] for i in idx]
            trees, _, _ = grow_forest(Xa, ya, n_trees=n_trees,
                                      min_leaf=min_leaf,
                                      seed=seed + int(arm))
            for i in val:
                w = forest_weights(trees, Xa, X[i])
                dest[i] = sum(w[t] * ya[t] for t in range(len(idx)))
        Xt, Wt = [X[i] for i in tr], [W[i] for i in tr]
        trees, _, _ = grow_forest(Xt, Wt, n_trees=n_trees,
                                  min_leaf=min_leaf, seed=seed + 7)
        for i in val:
            w = forest_weights(trees, Xt, X[i])
            e[i] = sum(w[t] * Wt[t] for t in range(len(tr)))
    return mu1, mu0, e


def aipw_scores(y, W, mu1, mu0, e, trim=0.02):
    r"""The augmented IPW score for each observation."""
    n = len(y)
    t = float(trim)
    if not 0.0 <= t < 0.5:
        raise ValueError("ipwgrf: trim must be in [0, 0.5), got %r"
                         % (trim,))
    g, weights = [], []
    for i in range(n):
        ei = min(max(float(e[i]), max(t, _EPS)), 1.0 - max(t, _EPS))
        wt = 1.0 / ei if W[i] == 1.0 else 1.0 / (1.0 - ei)
        weights.append(wt)
        g.append(mu1[i] - mu0[i]
                 + W[i] * (y[i] - mu1[i]) / ei
                 - (1.0 - W[i]) * (y[i] - mu0[i]) / (1.0 - ei))
    return g, weights


def ipw_forest(y, W, X, n_folds=5, n_trees=120, min_leaf=5, trim=0.02,
               seed=0, level=0.95, break_outcome=False,
               break_propensity=False):
    r"""ATE by an augmented, forest-fitted score.

    ``break_outcome`` and ``break_propensity`` deliberately misspecify
    one nuisance -- setting the outcome surfaces to a constant, or the
    propensity to the marginal -- so double robustness is something
    that can be demonstrated rather than asserted.
    """
    yv, Wv = k.vec(y), k.vec(W)
    n = len(yv)
    if len(Wv) != n:
        raise ValueError("ipwgrf: %d outcomes but %d treatments"
                         % (n, len(Wv)))
    if any(v not in (0.0, 1.0) for v in Wv):
        raise ValueError("ipwgrf: the treatment must be binary 0/1")
    if not 0 < sum(Wv) < n:
        raise ValueError("ipwgrf: both arms must be non-empty")
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("ipwgrf: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    if n < 60:
        raise ValueError("ipwgrf: need at least 60 observations, got %d"
                         % n)
    mu1, mu0, e = forest_nuisances(yv, Wv, Xm, n_folds=n_folds,
                                   n_trees=n_trees, min_leaf=min_leaf,
                                   seed=seed)
    if break_outcome:
        ybar = sum(yv) / n
        mu1 = [ybar] * n
        mu0 = [ybar] * n
    if break_propensity:
        e = [sum(Wv) / n] * n
    g, wts = aipw_scores(yv, Wv, mu1, mu0, e, trim=trim)
    psi = sum(g) / n
    se = k.sd(g) / math.sqrt(n) if n > 1 else float("nan")
    z = k.qnorm(0.5 + 0.5 * float(level))
    # the plug-in comparator, which corrects for nothing
    plug = sum(mu1[i] - mu0[i] for i in range(n)) / n
    return RichResult(payload={
        "estimate": psi, "ate": psi, "se": se,
        "ci": (psi - z * se, psi + z * se),
        "scores": g, "mu1": mu1, "mu0": mu0, "propensity": e,
        "plug_in": plug, "max_weight": max(wts),
        "min_propensity": min(e), "max_propensity": max(e),
        "trim": float(trim), "n": n, "level": float(level),
        "broken_outcome": bool(break_outcome),
        "broken_propensity": bool(break_propensity),
        "method": "augmented IPW with forest-fitted nuisances, Robins, "
                  "Rotnitzky & Zhao (1994) score, Athey, Tibshirani & "
                  "Wager (2019) forests",
    })


def cheatsheet():
    return ("ipwgrf: Gamma = mu1 - mu0 + W(Y-mu1)/e - (1-W)(Y-mu0)/"
            "(1-e), mean is the ATE. Right outcome model OR right "
            "propensity suffices, not neither. Cross-fit the nuisances, "
            "trim e, and report the largest weight -- a DR estimator "
            "with one weight of 500 is a one-observation estimator.")


# compact alias per ledger/NAMING.md
ipwforest = ipw_forest

# public names resolved by fn/_lazy_map.json
ipw_grf = ipw_forest
ipwgrf = ipw_forest
