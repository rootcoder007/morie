# morie.fn -- function file (rootcoder007/morie)
r"""Orthogonal Random Forest: local residualization, then a local fit.

The problem is a conditional moment with a nuisance in it:

.. math:: E\!\left[Y - \theta_0(x)T - f_0(x, W)\mid X = x\right] = 0,

where :math:`\theta_0(x)` is the heterogeneous effect of :math:`T` as a
function of the *few* features :math:`x` a decision maker is allowed to
act on, while :math:`W` is a possibly high-dimensional set of controls
that confound the assignment. The treatment policy
:math:`g_0(x, W) = E[T\mid x, W]` is unknown too.

**Neyman orthogonality is what makes the two-stage fit survive a slow
first stage.** Residualizing both sides,

.. math:: \tilde Y = Y - q_0(x, W), \qquad
          \tilde T = T - g_0(x, W), \qquad
          q_0(x,W) = E[Y \mid x, W],

gives the moment :math:`\psi = (\tilde Y - \theta \tilde T)\tilde T`,
whose derivative with respect to the nuisances vanishes at the truth.
The estimate of :math:`\theta` is then insensitive to first-order
nuisance error, which is why a first stage that converges slower than
:math:`n^{-1/2}` still leaves the second stage asymptotically normal.

**What separates ORF from a global residual-on-residual forest.** The
theory only needs the nuisance estimates to be accurate *near the
target point* :math:`x`. So ORF residualizes **locally**: the kernel
weights :math:`\alpha_i(x)` from a first-stage forest are used to fit
:math:`\hat q` and :math:`\hat g` in a neighbourhood of :math:`x`, and
those local residuals feed the second stage. Athey-Tibshirani-Wager's
"local centering" instead residualizes once, globally, and then runs
the forest on the residuals. The two coincide when the nuisance
functions do not vary with :math:`x`; they do not coincide otherwise,
and that difference is exactly what the anchor exercises. Both routes
are implemented -- ``residualize="local"`` and ``"global"`` -- because
the paper compares both and reports the gap.

**The second stage is a weighted least squares with no intercept.**
Given local residuals,

.. math:: \hat\theta(x) =
          \frac{\sum_i \alpha_i(x)\,\tilde T_i \tilde Y_i}
               {\sum_i \alpha_i(x)\,\tilde T_i^{2}},

which is the solution of the orthogonal moment under the forest
kernel. If the denominator is near zero there is no treatment
variation left near :math:`x` after residualizing, and the estimate is
refused rather than returned as a large number.

**Honesty and cross-fitting are not optional decoration.** The same
observations must not both choose the splits and supply the residuals,
or the second stage inherits the first stage's overfitting as bias.
The forest here is the honest one of :mod:`hntfst`, and the nuisance
fit at each point excludes the unit whose residual it is producing --
a leave-one-out that costs nothing at these sizes and removes the
self-prediction term exactly.

References
----------
Oprescu, M., Syrgkanis, V. & Wu, Z. S. (2019) "Orthogonal Random
Forest for Causal Inference", *Proceedings of the 36th International
Conference on Machine Learning*, PMLR 97, 4932-4941,
arXiv:1806.03467. Eq. (2) (the conditional moment), Sec. 1 (Neyman
orthogonality and why local residualization replaces global local
centering), Sec. 6 (the heterogeneous-treatment-effect application
implemented here).

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. & Robins, J. (2018) "Double/debiased machine learning for
treatment and structural parameters", *The Econometrics Journal*
21(1), C1-C68, doi:10.1111/ectj.12097. The residual-on-residual
construction and Neyman orthogonality that ORF localises.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized random
forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709. The forest weights alpha_i(x) and the
global "local centering" benchmark ORF is compared against.

Robinson, P. M. (1988) "Root-N-consistent semiparametric regression",
*Econometrica* 56(4), 931-954, doi:10.2307/1912705. The partially
linear model whose residual-on-residual estimator this generalises.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .hntfst import forest_weights, grow_forest

__all__ = ["local_nuisance", "orthogonal_moment", "orf_estimate",
           "orthogonal_random_forest"]

_EPS = 1e-10
_ROUTES = ("local", "global")


def _cols(A, n, name):
    M = k.mat(A)
    if len(M) != n:
        raise ValueError("orfgrf: %s has %d rows for %d observations"
                         % (name, len(M), n))
    return [[float(v) for v in r] for r in M]


def local_nuisance(target, W, weights, exclude=None, ridge=1e-8):
    r"""Weighted linear fit of ``target`` on ``W`` near a point.

    ``weights`` are the forest kernel weights :math:`\alpha_i(x)`, so
    the fit is local by construction: units far from :math:`x` in the
    forest's metric get no say. ``exclude`` drops one index, which is
    how the leave-one-out residual is formed.
    """
    n = len(target)
    Wm = k.mat(W)
    w = [float(v) for v in weights]
    if exclude is not None:
        w = list(w)
        w[int(exclude)] = 0.0
    if sum(w) <= _EPS:
        raise ValueError("orfgrf: the local neighbourhood is empty "
                         "after weighting")
    # k.wls prepends its own intercept column, so W is passed raw and
    # the returned coefficient vector is [intercept, slopes...].
    fit = k.wls(Wm, target, w, ridge=ridge)
    b = fit["coef"]
    return [b[0] + sum(Wm[i][j] * b[j + 1]
                       for j in range(len(b) - 1))
            for i in range(n)], b


def orthogonal_moment(y_res, t_res, weights):
    r"""Solve :math:`\sum_i \alpha_i \tilde T_i(\tilde Y_i -
    \theta\tilde T_i) = 0`.

    Refuses when the residualized treatment has no weighted variation
    left: that means the controls explain the treatment entirely near
    this point, and no effect is identified there.
    """
    n = len(y_res)
    if len(t_res) != n or len(weights) != n:
        raise ValueError("orfgrf: residuals and weights must agree in "
                         "length")
    den = sum(weights[i] * t_res[i] * t_res[i] for i in range(n))
    num = sum(weights[i] * t_res[i] * y_res[i] for i in range(n))
    scale = sum(weights[i] for i in range(n))
    if scale <= _EPS or den <= _EPS * max(scale, 1.0):
        raise ValueError("orfgrf: no residual treatment variation "
                         "near this point (weighted sum of T~^2 is "
                         "%.3g) -- the effect is not identified here"
                         % den)
    return num / den, den


def orf_estimate(Y, T, X, W, x, trees, residualize="local",
                 ridge=1e-8, leave_one_out=True):
    r"""One point estimate :math:`\hat\theta(x)`.

    ``residualize="local"`` fits the nuisances under the same kernel
    weights used for the second stage -- the ORF proposal.
    ``"global"`` fits them once on the whole sample, which is the
    "local centering" benchmark. They differ whenever the nuisance
    functions vary with :math:`x`.
    """
    if residualize not in _ROUTES:
        raise ValueError("orfgrf: residualize must be local or "
                         "global, got %r" % (residualize,))
    n = len(Y)
    w = forest_weights(trees, X, x)
    if residualize == "global":
        flat = [1.0 / n] * n
        qh, _ = local_nuisance(Y, W, flat, ridge=ridge)
        gh, _ = local_nuisance(T, W, flat, ridge=ridge)
        yr = [Y[i] - qh[i] for i in range(n)]
        tr = [T[i] - gh[i] for i in range(n)]
    elif leave_one_out:
        yr, tr = [], []
        for i in range(n):
            if w[i] <= 0.0:
                yr.append(0.0)
                tr.append(0.0)
                continue
            qh, _ = local_nuisance(Y, W, w, exclude=i, ridge=ridge)
            gh, _ = local_nuisance(T, W, w, exclude=i, ridge=ridge)
            yr.append(Y[i] - qh[i])
            tr.append(T[i] - gh[i])
    else:
        qh, _ = local_nuisance(Y, W, w, ridge=ridge)
        gh, _ = local_nuisance(T, W, w, ridge=ridge)
        yr = [Y[i] - qh[i] for i in range(n)]
        tr = [T[i] - gh[i] for i in range(n)]
    theta, den = orthogonal_moment(yr, tr, w)
    return theta, den, w


def orthogonal_random_forest(Y, T, X, W, x_eval=None, n_trees=100,
                             min_leaf=5, alpha=0.05, max_depth=12,
                             seed=0, residualize="local", ridge=1e-8,
                             kind="double-sample", leave_one_out=True):
    r"""ORF for the heterogeneous treatment effect :math:`\theta_0(x)`.

    Parameters
    ----------
    Y, T : array-like
        Outcome and treatment, length n. ``T`` may be continuous.
    X : array-like
        The features the effect is allowed to vary with -- typically
        few, because they are what a policy may act on.
    W : array-like
        The controls. May be high dimensional; they enter only through
        the nuisances and never through the splits.
    """
    y = [float(v) for v in k.vec(Y)]
    t = [float(v) for v in k.vec(T)]
    n = len(y)
    if len(t) != n:
        raise ValueError("orfgrf: %d treatments for %d outcomes"
                         % (len(t), n))
    Xm = _cols(X, n, "X")
    Wm = _cols(W, n, "W")
    if n < 8:
        raise ValueError("orfgrf: need at least 8 observations, got %d"
                         % n)
    trees, bags, s = grow_forest(Xm, y, W=t, kind=kind,
                                 n_trees=n_trees, min_leaf=min_leaf,
                                 alpha=alpha, max_depth=max_depth,
                                 seed=seed)
    pts = Xm if x_eval is None else k.mat(x_eval)
    thetas, dens = [], []
    for xx in pts:
        th, den, _ = orf_estimate(y, t, Xm, Wm, xx, trees,
                                  residualize=residualize, ridge=ridge,
                                  leave_one_out=leave_one_out)
        thetas.append(th)
        dens.append(den)
    return RichResult(payload={
        "estimate": sum(thetas) / len(thetas),
        "theta": thetas, "denominator": dens,
        "n": n, "n_trees": int(n_trees), "residualize": residualize,
        "n_controls": len(Wm[0]), "n_features": len(Xm[0]),
        "orthogonal": True,
        "method": "Orthogonal Random Forest, Oprescu, Syrgkanis & Wu "
                  "(2019), eq. (2) with %s residualization"
                  % residualize,
    })


def cheatsheet():
    return ("orfgrf: ORF. Moment E[Y - theta(x) T - f(x,W) | X=x] = 0. "
            "Residualize BOTH Y and T on the controls W, then "
            "theta(x) = sum a_i T~ Y~ / sum a_i T~^2 under forest "
            "weights. Neyman orthogonality means first-stage error "
            "enters only at second order. The ORF twist vs GRF local "
            "centering: residualize LOCALLY around x, not globally -- "
            "identical only when the nuisances do not vary with x.")


# compact alias per ledger/NAMING.md
orthogonalrandomforest = orthogonal_random_forest
