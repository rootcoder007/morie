# morie.fn -- shared engine (rootcoder007/morie)
"""Algorithm-agnostic variable importance: predictiveness and its gradient.

Implements the framework of Williamson, Gilbert, Simon and Carone
(2023), "A general framework for inference on algorithm-agnostic
variable importance", *JASA* 118:1645-1658 (preprint arXiv:2004.03683),
and the R-squared special case of Williamson, Gilbert, Carone and Simon
(2021), *Biometrics* 77:9-22.

The measures and their gradients
--------------------------------
A predictiveness measure ``V(f, P)`` scores a prediction function; the
importance of a variable group ``s`` is

    psi_s = V(f_full, P) - V(f_reduced, P),

where ``f_reduced`` is the best predictor that may not use ``s``. Both
terms are estimable, and the difference of their plug-ins needs NO
first-order debiasing correction: ``f_full`` maximises ``V(., P)``, so
the derivative of ``V`` with respect to the prediction function
vanishes at the optimum. That is the paper's central observation, and
it is why there is no fluctuation step here.

Every gradient below is mean-zero by construction. The gradients
printed in Appendix A of the JASA preprint are written up to an additive
constant -- the R-squared one has mean ``2v - 1`` rather than zero --
so the forms used here are the exact Gateaux derivatives, which agree
with the author's own ``vimp`` R package (``measure_r_squared.R``,
``measure_accuracy.R``, ``measure_auc.R``, ``measure_cross_entropy.R``).
:func:`gateaux_check` verifies each one numerically against a finite
difference.
"""

from . import _array_core as np

from ._trees_native import gb_fit, gb_predict

__all__ = [
    "MEASURES",
    "predictiveness",
    "gateaux_check",
    "fit_learner",
    "vim",
]

MEASURES = ("r_squared", "accuracy", "auc", "deviance")


def _as2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def predictiveness(y, pred, measure="r_squared", cutoff=0.5):
    """Point estimate of V(f, P_n) and its mean-zero gradient.

    Returns
    -------
    value : float
    grad : ndarray, shape (n,)
        The influence function of the plug-in, ``mean(grad) == 0`` to
        machine precision.
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    n = y.size
    if p.size != n:
        raise ValueError(
            "predictions and outcome disagree in length, %d vs %d."
            % (p.size, n)
        )
    if measure not in MEASURES:
        raise ValueError(
            "measure must be one of %s, got %r." % (MEASURES, measure)
        )
    if measure != "r_squared" and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError(
            "measure %r needs a binary 0/1 outcome." % measure
        )

    if measure == "r_squared":
        # v = 1 - MSE/Var. Both pieces have the elementary gradient
        # "squared error minus its own mean"; the delta method then
        # gives the ratio's gradient. vimp::measure_r_squared.R does
        # exactly this and it is what makes the result mean-zero.
        mse = float(np.mean((y - p) ** 2))
        mu = float(np.mean(y))
        var = float(np.mean((y - mu) ** 2))
        if var <= 0:
            raise ValueError(
                "the outcome has zero variance, so R-squared is undefined."
            )
        v = 1.0 - mse / var
        g_mse = (y - p) ** 2 - mse
        g_var = (y - mu) ** 2 - var
        grad = -(g_mse / var - mse * g_var / var ** 2)
        return v, grad

    if measure == "accuracy":
        correct = ((p > cutoff) == (y == 1)).astype(float)
        v = float(np.mean(correct))
        return v, correct - v

    if measure == "deviance":
        # v = 1 - CE/CE_null on the log-likelihood scale; both are
        # negative, so the ratio is the share of the null deviance the
        # model explains.
        pi = float(np.mean(y))
        if not 0.0 < pi < 1.0:
            raise ValueError(
                "the outcome is constant, so the null deviance is zero."
            )
        pc = np.clip(p, 1e-10, 1 - 1e-10)
        ll = y * np.log(pc) + (1 - y) * np.log(1 - pc)
        ce = float(np.mean(ll))
        lln = y * np.log(pi) + (1 - y) * np.log(1 - pi)
        den = float(np.mean(lln))
        v = 1.0 - ce / den
        grad = -((ll - ce) / den - ce * (lln - den) / den ** 2)
        return v, grad

    # AUC. The gradient splits by class: a control contributes the
    # share of cases it is correctly ranked below, a case the share of
    # controls it is correctly ranked above, each centred at the AUC
    # and scaled by its own class prevalence.
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC needs both classes present.")
    p1 = p[y == 1]
    p0 = p[y == 0]
    # ties count a half, the standard Mann-Whitney convention
    gt = np.sum(p1[:, None] > p0[None, :])
    eq = np.sum(p1[:, None] == p0[None, :])
    v = float((gt + 0.5 * eq) / (n1 * n0))
    sens = np.array([
        (np.sum(p0 < pi_) + 0.5 * np.sum(p0 == pi_)) / n0 for pi_ in p
    ])
    spec = np.array([
        (np.sum(p1 > pi_) + 0.5 * np.sum(p1 == pi_)) / n1 for pi_ in p
    ])
    pi1, pi0 = n1 / n, n0 / n
    grad = (y == 0) * (spec - v) / pi0 + (y == 1) * (sens - v) / pi1
    return v, grad


def gateaux_check(y, pred, measure="r_squared", n_points=40):
    """Numeric check that ``grad`` really is the Gateaux derivative.

    The tilt is applied EXACTLY rather than by sampling. Appending a
    second copy of observation :math:`i` to an :math:`n`-point sample
    produces the empirical distribution
    :math:`(1-\\epsilon)P_n + \\epsilon\\,\\delta_{z_i}` with
    :math:`\\epsilon = 1/(n+1)` and no Monte-Carlo error at all, so the
    finite difference :math:`(V_{aug} - V)/\\epsilon` differs from the
    analytic gradient only by the :math:`O(\\epsilon)` curvature term.
    A resampled tilt cannot do this job: at a usefully small
    :math:`\\epsilon` the sampling noise in :math:`V` divided by
    :math:`\\epsilon` swamps the gradient entirely.

    This is the check that catches a sign slip or a dropped
    delta-method term -- both of which leave a plausible-looking but
    wrong standard error rather than an obvious failure.

    Returns the maximum absolute discrepancy over the points examined.
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    n = y.size
    v0, grad = predictiveness(y, p, measure)
    eps = 1.0 / (n + 1.0)
    worst = 0.0
    for i in range(min(n, int(n_points))):
        vi, _ = predictiveness(
            np.append(y, y[i]), np.append(p, p[i]), measure
        )
        worst = max(worst, abs((vi - v0) / eps - grad[i]))
    return worst


def fit_learner(X, y, binary, n_estimators=150, max_depth=3,
                learning_rate=0.1):
    """Native gradient-boosted learner, returned as a callable.

    The theory needs an estimator of the oracle prediction function
    converging faster than :math:`n^{-1/4}`; boosted trees are a
    reasonable default, but the honest way to get one for a particular
    problem is to pass your own ``f`` to the public wrapper.
    """
    X = _as2d(X)
    if X.shape[1] == 0:
        m = float(np.mean(y))
        return lambda Z: np.full(np.asarray(Z).shape[0], m)
    task = "classification" if binary else "regression"
    fit = gb_fit(
        X, y, task=task, n_estimators=int(n_estimators),
        max_depth=int(max_depth), learning_rate=float(learning_rate),
    )
    return lambda Z: gb_predict(fit, _as2d(Z))


def vim(y, X, s, measure="r_squared", f=None, n_folds=5,
        sample_split=True, alpha=0.05, seed=0, **learner):
    r"""Variable importance for the group ``s``, with valid inference.

    Two distinct devices are in play and they are routinely confused.

    CROSS-FITTING (``n_folds``) trains the learner on one part of the
    data and evaluates the measure on another. Its job is to remove the
    Donsker condition, which would otherwise cap how flexible the
    learner may be.

    SAMPLE-SPLITTING (``sample_split``) estimates ``V(f_full)`` and
    ``V(f_reduced)`` on DISJOINT halves. Its job is entirely different:
    under the null of zero importance the two influence functions
    coincide, their difference is identically zero, and the plug-in has
    no non-degenerate limit -- so a Wald interval built on it has the
    wrong coverage no matter how large the sample. Splitting restores a
    non-degenerate limit and is the paper's stated recommendation for
    testing. It costs power, which is why it can be turned off when the
    importance is known to be non-null.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = _as2d(X)
    n = y.size
    if X.shape[0] != n:
        raise ValueError(
            "X has %d rows for %d outcomes." % (X.shape[0], n)
        )
    if n < 4 * int(n_folds):
        raise ValueError(
            "need at least 4 observations per fold, got n = %d for %d "
            "folds." % (n, int(n_folds))
        )
    s = np.atleast_1d(np.asarray(s, dtype=int)).ravel()
    if s.size == 0:
        raise ValueError("s is empty; name at least one column.")
    if s.min() < 0 or s.max() >= X.shape[1]:
        raise ValueError(
            "s refers to column %d, outside the %d columns of X."
            % (int(s.max()), X.shape[1])
        )
    keep = np.setdiff1d(np.arange(X.shape[1]), s)
    binary = bool(np.all(np.isin(y, (0.0, 1.0))))
    if measure != "r_squared" and not binary:
        raise ValueError("measure %r needs a binary outcome." % measure)

    rng = np.random.default_rng(int(seed))
    K = int(n_folds)

    def _learn(Xtr, ytr, cols):
        # f is a FITTING function, not a fitted one: the reduced-model
        # learner has to be trained without the dropped columns, and on
        # each fold's own training rows. Applying one pre-fitted f to
        # both blocks would estimate something else entirely.
        if f is not None:
            g = f(Xtr[:, cols], ytr)
            if not callable(g):
                raise TypeError(
                    "f must be a fitting function: f(X_train, y_train) has "
                    "to return a callable that predicts from new X."
                )
            return lambda Z: np.asarray(g(_as2d(Z))).ravel()
        return fit_learner(Xtr[:, cols], ytr, binary, **learner)

    def _crossfit(idx, cols):
        """Cross-fitted value and gradient over the rows in ``idx``."""
        folds = rng.permutation(idx.size) % K
        vals, etas = [], []
        for k in range(K):
            te = idx[folds == k]
            tr = idx[folds != k]
            if te.size < 2 or tr.size < 2:
                continue
            g = _learn(X[tr], y[tr], cols)
            v, grad = predictiveness(y[te], g(X[te][:, cols]), measure)
            vals.append(v)
            etas.append(float(np.mean(grad ** 2)))
        if not vals:
            raise ValueError("every fold was too small to evaluate.")
        return float(np.mean(vals)), float(np.mean(etas)), len(vals)

    full_cols = np.arange(X.shape[1])
    if sample_split:
        half = rng.permutation(n)
        a, b = half[: n // 2], half[n // 2:]
        v_full, eta_full, _ = _crossfit(a, full_cols)
        v_red, eta_red, _ = _crossfit(b, keep)
        var = eta_full / a.size + eta_red / b.size
        n_full, n_red = a.size, b.size
    else:
        v_full, eta_full, _ = _crossfit(np.arange(n), full_cols)
        v_red, eta_red, _ = _crossfit(np.arange(n), keep)
        # without splitting the two gradients are evaluated on the same
        # rows, so their difference -- not the sum of variances -- is
        # what the interval must be built from; the conservative
        # stand-in is used and flagged.
        var = (eta_full + eta_red) / n
        n_full = n_red = n

    psi = v_full - v_red
    se = float(np.sqrt(max(var, 0.0)))
    z = float(_z(1.0 - alpha / 2.0))
    z1 = float(_z(1.0 - alpha))
    return {
        "estimate": psi,
        "se": se,
        "ci": (psi - z * se, psi + z * se),
        "ci_one_sided": (psi - z1 * se, float("inf")),
        "test_statistic": psi / se if se > 0 else float("nan"),
        "p_value": float(_upper_tail(psi / se)) if se > 0 else float("nan"),
        "v_full": v_full,
        "v_reduced": v_red,
        "eta_full": eta_full,
        "eta_reduced": eta_red,
        "n_full": n_full,
        "n_reduced": n_red,
        "measure": measure,
        "s": s,
        "n_folds": K,
        "sample_split": bool(sample_split),
        "binary_outcome": binary,
        "n": n,
    }


def _z(q):
    """Standard normal quantile by bisection on the error function."""
    import math

    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _upper_tail(t):
    import math

    return 0.5 * math.erfc(t / math.sqrt(2.0))


def cheatsheet():
    return (
        "_vimp: predictiveness measures with their exact mean-zero "
        "gradients, cross-fitting for the Donsker condition and "
        "sample-splitting for the zero-importance null"
    )
