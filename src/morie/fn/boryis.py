# morie.fn -- function file (rootcoder007/morie)
"""Borusyak-Jaravel-Spiess imputation estimator."""

import numpy as np

from ._did import as_panel, first_treatment
from ._richresult import RichResult

__all__ = ["borusyak_jaravel_spiess", "impute_untreated"]


def _check_identified(obs, n, T):
    if obs.sum() < n + T - 1:
        raise ValueError(
            "only %d untreated cells for %d unit and period effects; the "
            "model is not identified. Every unit needs an untreated period "
            "and every period an untreated unit."
            % (int(obs.sum()), n + T - 1)
        )
    if not obs.any(axis=1).all():
        raise ValueError(
            "%d unit(s) are treated in every period, so their untreated "
            "level cannot be imputed." % int((~obs.any(axis=1)).sum())
        )
    if not obs.any(axis=0).all():
        raise ValueError(
            "%d period(s) have no untreated unit, so that period's effect "
            "cannot be identified." % int((~obs.any(axis=0)).sum())
        )


def _two_way_solve(obs, u_a, u_l, u_b, Xc, max_iter=2000, tol=1e-13):
    """Block Gauss-Seidel on the two-way normal equations over ``obs``.

    Solves ``(Z'Z) c = u`` where ``Z`` holds unit dummies, period
    dummies and any covariates, restricted to the observed cells.
    Iterating block means rather than forming ``Z`` keeps the memory
    at the size of the panel instead of (cells x parameters), which
    matters as soon as the panel has a few hundred units.
    """
    n, T = obs.shape
    n_i = obs.sum(axis=1).astype(float)
    m_t = obs.sum(axis=0).astype(float)
    a = np.zeros(n)
    lam = np.zeros(T)
    b = None if Xc is None else np.zeros(Xc.shape[2])
    XtX = None
    if Xc is not None:
        Z = Xc[obs]
        XtX = Z.T @ Z + 1e-12 * np.eye(Z.shape[1])
    for _ in range(int(max_iter)):
        a0, l0 = a.copy(), lam.copy()
        other = np.where(obs, lam[None, :], 0.0)
        if b is not None:
            other = other + np.where(obs, Xc @ b, 0.0)
        a = (u_a - other.sum(axis=1)) / n_i
        other = np.where(obs, a[:, None], 0.0)
        if b is not None:
            other = other + np.where(obs, Xc @ b, 0.0)
        lam = (u_l - other.sum(axis=0)) / m_t
        if b is not None:
            rest = np.where(obs, a[:, None] + lam[None, :], 0.0)
            b = np.linalg.solve(XtX, u_b - Xc[obs].T @ rest[obs])
        if max(np.max(np.abs(a - a0)), np.max(np.abs(lam - l0))) < tol:
            break
    return a, lam, b


def impute_untreated(Y, treated, X=None, max_iter=2000, tol=1e-13):
    r"""Fit unit and period effects on UNTREATED cells only and impute.

    With treated cells held out the panel is unbalanced, so the
    closed-form two-way demeaning does not apply and the effects come
    from the normal equations over the untreated cells. Returns the
    imputed :math:`\hat Y(0)` for every cell together with the fitted
    effects.
    """
    Y = np.asarray(Y, dtype=float)
    n, T = Y.shape
    obs = ~treated
    _check_identified(obs, n, T)
    Xc = None
    if X is not None:
        Xc = np.asarray(X, dtype=float)
        if Xc.ndim == 2:
            Xc = Xc[:, :, None]
    Yo = np.where(obs, Y, 0.0)
    u_b = None if Xc is None else Xc[obs].T @ Y[obs]
    a, lam, b = _two_way_solve(obs, Yo.sum(axis=1), Yo.sum(axis=0), u_b, Xc,
                               max_iter, tol)
    Y0 = a[:, None] + lam[None, :]
    if b is not None:
        Y0 = Y0 + Xc @ b
    return Y0, a, lam, b


def _estimator_weights(W, treated, Xc, max_iter=2000, tol=1e-13):
    r"""The exact linear weights ``v`` with ``tau_hat = sum(v * Y)``.

    The estimator subtracts a projection of the untreated outcomes,
    so it is linear in :math:`Y` and its weights can be written down:
    treated cells carry :math:`w_{it}`, and each untreated cell
    carries minus the amount by which it moves the imputation. Having
    :math:`v` makes the standard error a plain clustered sum instead
    of an approximation, and makes the identity
    :math:`\hat\tau = \sum v_{it} Y_{it}` checkable.
    """
    obs = ~treated
    u_a = W.sum(axis=1)
    u_l = W.sum(axis=0)
    u_b = None if Xc is None else np.einsum("ijk,ij->k", Xc, W)
    ca, cl, cb = _two_way_solve(obs, u_a, u_l, u_b, Xc, max_iter, tol)
    proj = ca[:, None] + cl[None, :]
    if cb is not None:
        proj = proj + Xc @ cb
    return np.where(treated, W, -proj * obs)


def borusyak_jaravel_spiess(y, D, unit, time, X=None, weights=None):
    r"""Impute the untreated potential outcome, then average the residuals.

    Borusyak, Jaravel and Spiess turn DiD around. Rather than picking
    a comparison group, fit the unit and period effects on the
    UNTREATED observations only,

    .. math:: \hat Y_{it}(0) = \hat\alpha_i + \hat\lambda_t
              + X_{it}'\hat\beta,

    then read each treated cell's effect straight off the residual:

    .. math:: \hat\tau_{it} = Y_{it} - \hat Y_{it}(0),
              \qquad
              \hat\tau = \sum_{it \in \mathcal{T}} w_{it}\hat\tau_{it}.

    Because no treated observation enters the fit, no already-treated
    unit can act as a control -- the contamination is ruled out by
    construction rather than by a weighting argument. It is also the
    efficient estimator under homoskedasticity, which the
    heterogeneity-robust alternatives are not.

    The cost is a stronger assumption: parallel trends must hold in
    EVERY pre-period, not only just before adoption, because the
    whole untreated panel is used to fit the effects. This function
    therefore returns ``pretrend_by_rel`` -- the mean residual at each
    pre-treatment relative period, which is zero in expectation under
    that assumption -- as a check rather than an afterthought.

    Parameters
    ----------
    y, D : array-like, shape (n,)
        Outcome and absorbing treatment in long format.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.
    X : array-like, shape (n, p), optional
        Time-varying covariates entering the imputation model.
    weights : {'equal', 'cohort'} or array-like, optional
        Weights over treated cells. ``'equal'`` (default) is the
        simple average over treated cells; an array is used directly.

    Returns
    -------
    RichResult
        ``estimate``, ``se`` (unit-clustered), ``ci``, ``tau_it``,
        ``event`` (mean effect by relative period), ``cohort_att``,
        ``pretrend_by_rel``, ``pretrend_max_abs``, ``unit_effects``,
        ``period_effects``, ``n_treated_cells``.

    References
    ----------
    Borusyak, Jaravel and Spiess (2024), *Review of Economic Studies*
    91:3253-3285.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(9), 8)
    >>> time = np.tile(np.arange(8), 9)
    >>> gv = np.repeat([3., 3., 3., 5., 5., 5., np.inf, np.inf, np.inf], 8)
    >>> D = (time >= gv).astype(float)
    >>> y = unit * 0.3 + time * 0.2 + 2.0 * D
    >>> round(borusyak_jaravel_spiess(y, D, unit, time)["estimate"], 10)
    2.0
    """
    Y, units, periods = as_panel(y, unit, time)
    g, Dm, _, _ = first_treatment(D, unit, time, units, periods)
    treated = Dm > 0
    if not treated.any():
        raise ValueError("no observation is treated.")
    Xp = None
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        Xp = np.stack([as_panel(Xa[:, j], unit, time)[0]
                       for j in range(Xa.shape[1])], axis=2)

    Y0, alpha, lam, beta = impute_untreated(Y, treated, Xp)
    tau = Y - Y0
    n_u, T = Y.shape

    if weights is None or (isinstance(weights, str) and weights == "equal"):
        W = treated.astype(float)
    elif isinstance(weights, str) and weights == "cohort":
        W = np.zeros_like(Y)
        for gg in np.unique(g[np.isfinite(g)]):
            rows = g == gg
            W[rows] = treated[rows] * (rows.sum() / float(n_u))
        W = W * treated
    else:
        Wa = np.asarray(weights, dtype=float)
        W = as_panel(Wa, unit, time)[0] if Wa.size == len(y) else Wa
        W = np.where(treated, W, 0.0)
        if W.sum() <= 0:
            raise ValueError("the supplied weights put no mass on treated cells.")
    W = W / W.sum()
    est = float(np.sum(W * tau))

    # exact linear weights, then a unit-clustered sum over them. The
    # residual is the untreated fit's residual where one exists, and the
    # treated cell's deviation from the average effect where it does not
    # -- the conservative choice, since that deviation is real
    # heterogeneity as well as noise.
    v = _estimator_weights(W, treated, Xp)
    e = np.where(treated, tau - est, Y - Y0)
    scores = np.sum(v * e, axis=1)
    se = float(np.sqrt(n_u / max(n_u - 1.0, 1.0) * np.sum(scores**2)))
    linearity_residual = float(np.sum(v * Y) - est)

    rel = np.full(Y.shape, np.nan)
    fin = np.isfinite(g)
    rel[fin] = np.arange(T)[None, :] - g[fin][:, None]
    event, pre = {}, {}
    for r in np.unique(rel[~np.isnan(rel)]):
        m = (rel == r)
        if r >= 0:
            event[float(r)] = float(tau[m].mean())
        else:
            pre[float(r)] = float(tau[m].mean())
    cohort_att = {
        float(gg): float(tau[(g == gg)[:, None] & treated].mean())
        for gg in np.unique(g[fin])
        if treated[g == gg].any()
    }

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - z * se, est + z * se),
            "tau_it": tau,
            "imputed_y0": Y0,
            "weights": v,
            "linearity_residual": linearity_residual,
            "se_note": (
                "clustered on unit using the estimator's exact linear "
                "weights; treated cells contribute their deviation from the "
                "average effect, which is the conservative choice"
            ),
            "event": event,
            "cohort_att": cohort_att,
            "pretrend_by_rel": pre,
            "pretrend_max_abs": float(max((abs(v) for v in pre.values()),
                                          default=0.0)),
            "pretrend_note": (
                "the imputation uses EVERY untreated period, so parallel "
                "trends is assumed throughout the pre-period, not only just "
                "before adoption; these residuals are zero in expectation "
                "under that assumption"
            ),
            "unit_effects": alpha,
            "period_effects": lam,
            "covariate_coef": beta,
            "n_treated_cells": int(treated.sum()),
            "n_untreated_cells": int((~treated).sum()),
            "n_units": int(n_u),
            "n_periods": int(T),
            "no_forbidden_comparisons": (
                "no treated observation enters the fit, so an already-treated "
                "unit cannot act as a control"
            ),
            "method": "Borusyak-Jaravel-Spiess (2024) imputation estimator",
        }
    )


def cheatsheet():
    return (
        "boryis: fit unit and period effects on untreated cells, impute "
        "Y(0), average the treated residuals; efficient, with a full "
        "pre-period parallel-trends check"
    )
