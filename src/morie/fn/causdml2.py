# morie.fn -- function file (rootcoder007/morie)
"""Double machine learning for the partially linear model."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_dml_partial_lin"]


def _ridge_learner(alpha=1.0):
    def fit(Xtr, ytr):
        Xc = np.column_stack([np.ones(len(ytr)), Xtr])
        p = Xc.shape[1]
        pen = alpha * np.eye(p)
        pen[0, 0] = 0.0
        b = np.linalg.solve(Xc.T @ Xc + pen, Xc.T @ ytr)
        return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b
    return fit


def causal_dml_partial_lin(y, D, X, n_folds=5, learner=None, seed=0):
    r"""Double machine learning for the partially linear model
    :math:`Y = \theta D + g(X) + \varepsilon`,
    :math:`D = m(X) + \nu`, estimated by

    .. math:: \hat\theta = \left(\tilde D'\tilde D\right)^{-1}
              \tilde D'\tilde Y,

    with :math:`\tilde Y = Y - \hat\ell(X)` and
    :math:`\tilde D = D - \hat m(X)` **cross-fitted**: each
    observation's nuisance prediction comes from a model fitted
    WITHOUT it.

    Two ingredients make this work and neither is optional.

    The score is NEYMAN-ORTHOGONAL. Residualising both :math:`Y` and
    :math:`D` on :math:`X` makes the moment condition insensitive to
    first-order errors in the nuisance estimates, so a slowly
    converging :math:`\hat g` does not contaminate :math:`\hat\theta`
    at the same rate. Residualising only :math:`Y` -- regressing the
    outcome residual on raw :math:`D` -- is NOT orthogonal and
    reintroduces the bias.

    The nuisances are CROSS-FITTED. Fitting them on the same data
    used for the final moment leaves a regularisation bias that does
    not vanish at :math:`\sqrt n`; the fold split is what removes it.
    ``cross_fitted`` is always ``True`` here, and
    ``theta_in_sample`` is computed alongside precisely so the two
    can be compared -- on a design with many covariates the gap is
    large and visible, which is the point of the method.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    d : array-like, shape (n,)
        Treatment, continuous or binary.
    x : array-like, shape (n, p)
        Controls.
    n_folds : int, default 5
        Cross-fitting folds; at least 2.
    learner : callable, optional
        ``learner(X_train, y_train)`` returning a predict callable.
        A ridge fit when omitted.
    seed : int, default 0
        Fold-assignment seed.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se``, ``ci``, ``y_residual``,
        ``d_residual``, ``theta_in_sample``, ``cross_fitted``,
        ``n_folds``, ``first_stage_r2``, ``n``, ``p``, ``method``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
    Hansen, C., Newey, W. and Robins, J. (2018), "Double/debiased
    machine learning for treatment and structural parameters",
    *Econometrics Journal* 21:C1-C68.
    """
    from ._caus_iv import folds

    yv = np.asarray(y, dtype=float).ravel()
    Dv = np.asarray(D, dtype=float).ravel()
    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    if Xm.shape[0] != yv.size:
        Xm = Xm.T
    n, p = Xm.shape
    if not (Dv.size == yv.size == n):
        raise ValueError("y, D and X must agree on the number of rows.")
    fit = _ridge_learner() if learner is None else learner
    fs = folds(n, n_folds, seed=seed)

    yres = np.empty(n)
    dres = np.empty(n)
    for te in fs:
        tr = np.setdiff1d(np.arange(n), te)
        yres[te] = yv[te] - fit(Xm[tr], yv[tr])(Xm[te])
        dres[te] = Dv[te] - fit(Xm[tr], Dv[tr])(Xm[te])
    den = float(dres @ dres)
    if den <= 0:
        raise ValueError(
            "the residualised treatment has no variation left: X explains D "
            "completely, so theta is not identified.")
    theta = float(dres @ yres / den)
    eps = yres - theta * dres
    se = float(np.sqrt(np.sum(dres ** 2 * eps ** 2)) / den)

    # the same estimator WITHOUT cross-fitting, for comparison only
    yr_in = yv - fit(Xm, yv)(Xm)
    dr_in = Dv - fit(Xm, Dv)(Xm)
    din = float(dr_in @ dr_in)
    theta_in = float(dr_in @ yr_in / din) if din > 0 else np.nan
    return RichResult(payload={
        "theta": theta, "se": se,
        "ci": (theta - 1.959963984540054 * se,
               theta + 1.959963984540054 * se),
        "y_residual": yres, "d_residual": dres,
        "theta_in_sample": theta_in,
        "cross_fitted": True, "n_folds": int(len(fs)),
        "first_stage_r2": float(1.0 - np.var(dres) / np.var(Dv))
        if np.var(Dv) > 0 else np.nan,
        "why_cross_fit": "fitting the nuisances on the data used for the "
                         "final moment leaves a regularisation bias that "
                         "does not vanish at root-n; theta_in_sample is "
                         "reported so the gap is visible",
        "why_orthogonal": "residualising BOTH Y and D on X makes the score "
                          "Neyman-orthogonal; residualising only Y is not "
                          "orthogonal and the bias returns",
        "n": int(n), "p": int(p),
        "method": "Double machine learning, partially linear model (Chernozhukov et al. 2018)"})


def cheatsheet():
    return "causdml2: residualise BOTH Y and D, and cross-fit -- either omission brings the bias back"
