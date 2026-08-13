# morie.fn -- function file (rootcoder007/morie)
r"""Partial-linear GRF: heterogeneous effects with high-dimensional controls.

The partially linear model

.. math:: Y_i = \tau(X_i)\,(W_i - e(X_i)) + m(X_i) + \varepsilon_i,
          \qquad e(x) = E[W \mid X=x],\ m(x) = E[Y \mid X=x],

separates what is being estimated, :math:`\tau(\cdot)`, from the
nuisance surfaces :math:`m` and :math:`e` that merely have to be
controlled for.

**Local centering is what makes it work, and skipping it is the usual
mistake.** Residualise first -- :math:`\tilde Y = Y - \hat m(X)` and
:math:`\tilde W = W - \hat e(X)` -- and only then fit the forest. The
GRF estimating equation (2) with weights (3) then solves

.. math:: \hat\tau(x) = \frac{\sum_i \alpha_i(x)\,\tilde W_i\,\tilde
          Y_i}{\sum_i \alpha_i(x)\,\tilde W_i^2},

which is a weighted Robinson regression run in the forest's own
neighbourhood. Without the centering, the forest spends its splits
chasing variation in :math:`m(X)` -- the confounding surface -- instead
of variation in the treatment effect, and the estimate absorbs the
confounder. The anchor builds a design where :math:`m` is strong and
:math:`\tau` is weak, which is exactly the case that separates them.

**Orthogonality is the reason the nuisances may be estimated roughly.**
The score :math:`\psi_\tau = (\tilde Y - \tau\tilde W)\tilde W` has zero
derivative in both nuisances at the truth, so first-order errors in
:math:`\hat m` and :math:`\hat e` cancel rather than propagate -- the
R-learner property.

References
----------
Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized Random
Forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709, arXiv:1610.01271. Eq. (2), (3), (8); Sec. 6.1.1
on local centering.

Nie, X. & Wager, S. (2021) "Quasi-oracle estimation of heterogeneous
treatment effects", *Biometrika* 108(2), 299-319,
doi:10.1093/biomet/asaa076. The R-learner objective this solves
locally.

Robinson, P. M. (1988) "Root-N-Consistent Semiparametric Regression",
*Econometrica* 56(4), 931-954, doi:10.2307/1912705. The partially
linear model and the residual-on-residual construction.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. & Robins, J. (2018) "Double/debiased machine learning for
treatment and structural parameters", *The Econometrics Journal* 21(1),
C1-C68, doi:10.1111/ectj.12097. Neyman orthogonality and cross-fitting.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest

__all__ = ["partial_linear_grf", "local_centering", "residual_forest"]

_EPS = 1e-12


def _folds(n, V):
    V = max(2, min(int(V), n))
    return [[i for i in range(n) if i % V == v] for v in range(V)]


def local_centering(y, W, X, n_folds=5, n_trees=100, min_leaf=5,
                    seed=0):
    r"""Cross-fitted :math:`\hat m(X)` and :math:`\hat e(X)`.

    Cross-fitted because a nuisance fitted on the same rows it is
    subtracted from leaves a residual correlated with its own error,
    which is the overfitting bias orthogonality does NOT protect
    against.
    """
    n = len(y)
    mh, eh = [0.0] * n, [0.0] * n
    for val in _folds(n, n_folds):
        tr = [i for i in range(n) if i not in set(val)]
        if not tr:
            continue
        fm = _forest_predict(X, y, tr, val, n_trees, min_leaf, seed)
        fe = _forest_predict(X, W, tr, val, n_trees, min_leaf, seed + 1)
        for t, i in enumerate(val):
            mh[i], eh[i] = fm[t], fe[t]
    return mh, eh


def _forest_predict(X, y, train, at_rows, n_trees, min_leaf, seed):
    trees, _, _ = grow_forest([X[i] for i in train],
                              [y[i] for i in train], n_trees=n_trees,
                              min_leaf=min_leaf, seed=seed)
    Xt = [X[i] for i in train]
    out = []
    for i in at_rows:
        w = forest_weights(trees, Xt, X[i])
        out.append(sum(w[t] * y[train[t]] for t in range(len(train))))
    return out


def residual_forest(y_res, w_res, X, at=None, n_trees=200, min_leaf=5,
                    seed=0, alpha=0.05, pi=0.5):
    r"""Solve eq. (2) in the forest neighbourhood: the weighted
    Robinson ratio at each point."""
    n = len(y_res)
    # the forest is grown on the residual outcome, so its splits chase
    # variation in tau rather than in m
    trees, bags, s = grow_forest(X, y_res, n_trees=n_trees,
                                 min_leaf=min_leaf, seed=seed,
                                 alpha=alpha, pi=pi)
    Q = k.mat(at) if at is not None else X
    tau, num, den = [], [], []
    for q in range(len(Q)):
        a = forest_weights(trees, X, Q[q])
        wbar = sum(a[i] * w_res[i] for i in range(n))
        ybar = sum(a[i] * y_res[i] for i in range(n))
        nu = sum(a[i] * (w_res[i] - wbar) * (y_res[i] - ybar)
                 for i in range(n))
        de = sum(a[i] * (w_res[i] - wbar) ** 2 for i in range(n))
        if abs(de) < _EPS:
            raise ValueError("plrgrf: no treatment variation in the "
                             "neighbourhood of a target point")
        tau.append(nu / de)
        num.append(nu)
        den.append(de)
    return tau, {"trees": trees, "bags": bags, "s": s,
                 "numerator": num, "denominator": den}


def partial_linear_grf(y, W, X, at=None, n_trees=200, n_folds=5,
                       min_leaf=5, seed=0, center=True, level=0.95):
    r"""CATE by partial-linear GRF.

    ``center=False`` skips the residualisation. It exists because the
    confounding it lets through is the point -- a bias you can measure
    beats an instruction to remember.
    """
    yv, Wv = k.vec(y), k.vec(W)
    n = len(yv)
    if len(Wv) != n:
        raise ValueError("plrgrf: %d outcomes but %d treatments"
                         % (n, len(Wv)))
    Xm = k.mat(X)
    if len(Xm) != n:
        raise ValueError("plrgrf: %d covariate rows for %d outcomes"
                         % (len(Xm), n))
    if n < 40:
        raise ValueError("plrgrf: need at least 40 observations, got %d"
                         % n)
    if center:
        mh, eh = local_centering(yv, Wv, Xm, n_folds=n_folds,
                                 n_trees=max(50, n_trees // 2),
                                 min_leaf=min_leaf, seed=seed)
    else:
        mh = [0.0] * n
        eh = [0.0] * n
    yr = [yv[i] - mh[i] for i in range(n)]
    wr = [Wv[i] - eh[i] for i in range(n)]
    tau, info = residual_forest(yr, wr, Xm, at=at, n_trees=n_trees,
                                min_leaf=min_leaf, seed=seed)
    Q = k.mat(at) if at is not None else Xm
    # influence-function standard error for the weighted ratio
    ses = []
    for q in range(len(Q)):
        a = forest_weights(info["trees"], Xm, Q[q])
        wbar = sum(a[i] * wr[i] for i in range(n))
        de = info["denominator"][q]
        psi = [a[i] * (wr[i] - wbar) * (yr[i] - tau[q] * wr[i])
               for i in range(n)]
        v = sum(p * p for p in psi) / (de * de) if de else float("nan")
        ses.append(math.sqrt(max(v, 0.0)))
    z = k.qnorm(0.5 + 0.5 * float(level))
    return RichResult(payload={
        "estimate": tau, "tau": tau, "se": ses,
        "ci": [(tau[q] - z * ses[q], tau[q] + z * ses[q])
               for q in range(len(Q))],
        "m_hat": mh, "e_hat": eh, "y_residual": yr, "w_residual": wr,
        "centered": bool(center), "n": n, "n_trees": int(n_trees),
        "ate": sum(tau) / len(tau), "level": float(level),
        "method": "partial-linear generalized random forest, Athey, "
                  "Tibshirani & Wager (2019) eq. (2)-(3) with local "
                  "centering",
    })


def cheatsheet():
    return ("plrgrf: residualise FIRST -- Ytilde = Y - m(X), Wtilde = "
            "W - e(X), both cross-fitted -- then solve eq. (2) in the "
            "forest neighbourhood: tau(x) = sum a_i Wtilde Ytilde / "
            "sum a_i Wtilde^2. Skip the centering and the forest splits "
            "on m(X), the confounding surface, not on tau.")


# compact alias per ledger/NAMING.md
partiallineargrf = partial_linear_grf
