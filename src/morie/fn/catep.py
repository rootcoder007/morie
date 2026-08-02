# morie.fn -- function file (rootcoder007/morie)
"""CATE estimation by S-, T-, X- and R-learner meta-algorithms."""

from . import _array_core as np

from ._cforest import CausalForest
from ._did import add_intercept, logit_fit, logit_predict, ols_fit
from ._richresult import RichResult

__all__ = ["cate_estimation"]

_ESTIMATORS = ("t", "s", "x", "r", "forest")


def _basis(X, degree):
    """Polynomial basis, so the linear base learners can bend."""
    cols = [np.ones(X.shape[0])]
    for j in range(X.shape[1]):
        for d in range(1, degree + 1):
            cols.append(X[:, j] ** d)
    return np.column_stack(cols)


def cate_estimation(Y, T, X, estimator="x", degree=2, n_trees=200, seed=0):
    r"""Conditional average treatment effects from a chosen meta-learner.

    :math:`\tau(x) = E[Y(1) - Y(0) \mid X = x]` is never observed for
    anyone, so every method here is a recipe for combining regressions
    that ARE estimable. The recipes differ in what they assume, and the
    differences matter more than the choice of base learner:

    ``'s'`` (single)
        One regression on :math:`(X, T)` jointly, then differencing
        :math:`T`. Efficient when the effect is small or absent,
        because nothing forces the model to spend structure on
        :math:`T` -- and that is also its failure: a flexible learner
        can drop :math:`T` entirely and return :math:`\hat\tau \equiv 0`.

    ``'t'`` (two)
        Separate regressions on the treated and controls. Makes no
        assumption that the arms share structure, which costs
        precision when they do, and breaks down when one arm is small
        -- its model is fitted on the few observations it has.

    ``'x'`` (cross, default)
        Impute each unit's missing potential outcome from the OTHER
        arm's model, giving two effect estimates, then blend them with
        propensity weights :math:`\hat\tau = g\hat\tau_0 +
        (1-g)\hat\tau_1`. The weighting is the point: where treated
        units are scarce, the estimate leans on the control arm's
        larger model. Künzel et al. designed it for exactly the
        unbalanced case that defeats the T-learner.

    ``'r'`` (residual)
        Regress the outcome residual on the treatment residual,
        :math:`\tilde Y \approx \tau(x)\tilde T`. Robinson's
        transformation makes the estimate insensitive to first-order
        errors in the nuisance fits -- the Neyman-orthogonality
        property the other three lack.

    ``'forest'``
        The honest causal forest, for when :math:`\tau` is not smooth.

    All five are returned in ``by_estimator``, because agreement
    between them is the only cheap evidence that the heterogeneity is
    real rather than an artefact of one recipe.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    T : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    estimator : {'x', 't', 's', 'r', 'forest'}
        Which learner supplies ``cate``.
    degree : int
        Polynomial degree for the linear base learners.
    n_trees, seed :
        Forest controls for ``estimator='forest'``.

    Returns
    -------
    RichResult
        ``cate``, ``uncertainty`` (spread across the five learners at
        each point), ``ate``, ``by_estimator``, ``agreement``
        (correlation matrix), ``estimator``.

    References
    ----------
    Künzel, Sekhon, Bickel and Yu (2019), *PNAS* 116:4156-4165
    (S-, T- and X-learners).
    Nie and Wager (2021), *Biometrika* 108:299-319 (the R-learner).
    Robinson (1988), *Econometrica* 56:931-954.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(2000, 2))
    >>> T = (rng.uniform(size=2000) < 0.5).astype(float)
    >>> Y = X[:, 0] + T * (1 + X[:, 0]) + rng.normal(scale=0.3, size=2000)
    >>> out = cate_estimation(Y, T, X, estimator="x")
    >>> bool(abs(out["ate"] - 1.0) < 0.15)
    True
    """
    if estimator not in _ESTIMATORS:
        raise ValueError(
            "estimator must be one of %s, got %r." % (_ESTIMATORS, estimator)
        )
    y = np.asarray(Y, dtype=float).ravel()
    t = np.asarray(T, dtype=float).ravel()
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    n = y.size
    if not (t.size == n == Xa.shape[0]):
        raise ValueError(
            "Y, T and X must agree in length, got %d, %d and %d."
            % (n, t.size, Xa.shape[0])
        )
    if not np.all(np.isin(t, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    n1, n0 = int(t.sum()), int((1 - t).sum())
    if n1 < 5 or n0 < 5:
        raise ValueError(
            "need at least 5 units in each arm, got %d treated and %d "
            "control." % (n1, n0)
        )

    B = _basis(Xa, int(degree))
    tr, ct = t == 1, t == 0
    g = logit_predict(add_intercept(Xa), logit_fit(add_intercept(Xa), t)[0])
    g = np.clip(g, 0.02, 0.98)

    out = {}

    # S-learner: one model, with T and its interactions
    Bs = np.column_stack([B, t[:, None] * B])
    cs = ols_fit(Bs, y)
    out["s"] = (np.column_stack([B, B]) @ cs) - (
        np.column_stack([B, 0 * B]) @ cs
    )

    # T-learner: one model per arm
    b1 = ols_fit(B[tr], y[tr])
    b0 = ols_fit(B[ct], y[ct])
    mu1, mu0 = B @ b1, B @ b0
    out["t"] = mu1 - mu0

    # X-learner: impute the missing arm, then blend by propensity
    d1 = y[tr] - mu0[tr]
    d0 = mu1[ct] - y[ct]
    tau1 = B @ ols_fit(B[tr], d1)
    tau0 = B @ ols_fit(B[ct], d0)
    out["x"] = g * tau0 + (1 - g) * tau1

    # R-learner: residual-on-residual (Robinson transformation)
    yt, tt = y - B @ ols_fit(B, y), t - g
    Bw = B * tt[:, None]
    # normal equations of min sum (Ytilde - tau(x) Ttilde)^2 with
    # tau(x) = B(x)'theta: (B Ttilde)'(B Ttilde) theta = (B Ttilde)' Ytilde.
    # The outcome residual enters ONCE -- weighting it by Ttilde again
    # solves a different problem and returns a near-zero effect.
    out["r"] = B @ np.linalg.solve(
        Bw.T @ Bw + 1e-10 * np.eye(B.shape[1]), Bw.T @ yt
    )

    forest = CausalForest(n_trees=int(n_trees), seed=int(seed)).fit(Xa, y, t)
    out["forest"] = forest.predict()

    M = np.column_stack([out[k] for k in _ESTIMATORS])
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(M, rowvar=False)
    return RichResult(
        payload={
            "cate": out[estimator],
            "estimate": float(np.mean(out[estimator])),
            "ate": float(np.mean(out[estimator])),
            "uncertainty": M.std(axis=1),
            "uncertainty_note": (
                "spread ACROSS learners at each point, not a standard error; "
                "it says how much the answer depends on the recipe"
            ),
            "by_estimator": {k: out[k] for k in _ESTIMATORS},
            "ate_by_estimator": {k: float(np.mean(out[k]))
                                 for k in _ESTIMATORS},
            "agreement": corr,
            "agreement_note": (
                "learners that disagree about the SHAPE of tau(x) are the "
                "cheapest available evidence that the heterogeneity is an "
                "artefact of one recipe"
            ),
            "estimator": estimator,
            "propensity_range": (float(g.min()), float(g.max())),
            "degree": int(degree),
            "n_treated": n1,
            "n_control": n0,
            "n": n,
            "method": "CATE by %s-learner" % estimator,
        }
    )


def cheatsheet():
    return (
        "catep: CATE by S-, T-, X-, R-learner or causal forest, all five "
        "returned so their disagreement is visible"
    )
