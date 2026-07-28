# morie.fn -- function file (rootcoder007/morie)
"""Wooldridge's extended two-way fixed-effects estimator."""

import numpy as np

from ._did import as_panel, first_treatment
from ._richresult import RichResult

__all__ = ["wooldridge_bjs_estimator"]


def wooldridge_bjs_estimator(y, D, unit, time, X=None):
    r"""Extended TWFE: one interaction per cohort-period cell.

    Wooldridge's (2021) point is that TWFE is not wrong because it is
    a regression; it is wrong because it is the WRONG regression. The
    usual specification forces a single coefficient :math:`\beta` on
    :math:`D_{it}`, which imposes that every cohort has the same
    effect at every horizon. Saturate that restriction away --

    .. math:: y_{it} = \alpha_i + \lambda_t
              + \sum_{g}\sum_{s \geq g} \tau_{gs}\,
                \mathbb{1}\{G_i = g\}\,\mathbb{1}\{t = s\}
              + X_{it}'\gamma + \varepsilon_{it}

    -- and the coefficients :math:`\hat\tau_{gs}` are exactly the
    :math:`ATT(g,s)` estimates, with no pooling and therefore no
    negative weights. Aggregating them is then the user's explicit
    choice rather than something the regression did silently.

    Under this saturation the estimator is numerically IDENTICAL to
    the Borusyak-Jaravel-Spiess imputation estimator without
    covariates: both fit the untreated cells and read the treated
    residuals. ``matches_imputation`` records the largest discrepancy
    on the data at hand, so the equivalence is demonstrated rather
    than asserted.

    Parameters
    ----------
    y, D : array-like, shape (n,)
        Outcome and absorbing treatment in long format.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.
    X : array-like, shape (n, p), optional
        Time-varying covariates entering with a common coefficient.

    Returns
    -------
    RichResult
        ``estimate`` (equally weighted over treated cells), ``se``,
        ``ci``, ``att_gt``, ``event``, ``cohort_att``,
        ``matches_imputation``, ``n_interactions``, ``covariate_coef``.

    Notes
    -----
    The saturated design has one parameter per treated cohort-period
    cell. With many cohorts and periods that is a lot of parameters
    and each is estimated from few observations; ``min_cell_size``
    reports the thinnest one, since a cohort-period cell of size 1 is
    a coefficient with no residual degrees of freedom.

    References
    ----------
    Wooldridge (2021), "Two-Way Fixed Effects, the Two-Way Mundlak
    Regression, and Difference-in-Differences Estimators", SSRN
    3906345.
    Borusyak, Jaravel and Spiess (2024), *ReStud* 91:3253-3285.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(9), 8)
    >>> time = np.tile(np.arange(8), 9)
    >>> gv = np.repeat([3., 3., 3., 5., 5., 5., np.inf, np.inf, np.inf], 8)
    >>> D = (time >= gv).astype(float)
    >>> y = unit * 0.3 + time * 0.2 + 2.0 * D
    >>> round(wooldridge_bjs_estimator(y, D, unit, time)["estimate"], 10)
    2.0
    """
    Y, units, periods = as_panel(y, unit, time)
    g, Dm, _, _ = first_treatment(D, unit, time, units, periods)
    treated = Dm > 0
    if not treated.any():
        raise ValueError("no observation is treated.")
    n_u, T = Y.shape

    cells = [(float(gg), int(t))
             for gg in np.unique(g[np.isfinite(g)])
             for t in range(int(gg), T)
             if ((g == gg) & treated[:, t]).any()]
    if not cells:
        raise ValueError("no cohort-period cell is treated.")

    # unit dummies (drop none), period dummies (drop first), one
    # interaction per treated cohort-period cell, then covariates
    n_obs = n_u * T
    cols = [np.zeros((n_u, T)) for _ in range(n_u + (T - 1) + len(cells))]
    for i in range(n_u):
        cols[i][i, :] = 1.0
    for t in range(1, T):
        cols[n_u + t - 1][:, t] = 1.0
    for j, (gg, t) in enumerate(cells):
        cols[n_u + T - 1 + j][(g == gg), t] = 1.0
    Z = np.column_stack([c.ravel() for c in cols])
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        Xp = np.column_stack([as_panel(Xa[:, j], unit, time)[0].ravel()
                              for j in range(Xa.shape[1])])
        Z = np.column_stack([Z, Xp])
    yv = Y.ravel()
    coef, *_ = np.linalg.lstsq(Z, yv, rcond=None)
    resid = yv - Z @ coef

    k = n_u + T - 1
    att = {cells[j]: float(coef[k + j]) for j in range(len(cells))}
    n_cell = {c: int(((g == c[0]) & treated[:, c[1]]).sum()) for c in cells}
    tot = sum(n_cell.values())
    est = float(sum(n_cell[c] / tot * att[c] for c in cells))

    # unit-clustered covariance for the interaction block
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    Rm = resid.reshape(n_u, T)
    Zm = Z.reshape(n_u, T, Z.shape[1])
    meat = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(n_u):
        s = Zm[i].T @ Rm[i]
        meat += np.outer(s, s)
    V = ZtZ_inv @ meat @ ZtZ_inv * (n_u / max(n_u - 1.0, 1.0))
    w = np.zeros(Z.shape[1])
    for j, c in enumerate(cells):
        w[k + j] = n_cell[c] / tot
    se = float(np.sqrt(max(w @ V @ w, 0.0)))

    event, cohort_att = {}, {}
    for (gg, t), a in att.items():
        r = float(t - gg)
        event.setdefault(r, []).append((n_cell[(gg, t)], a))
        cohort_att.setdefault(gg, []).append((n_cell[(gg, t)], a))
    event = {r: float(sum(n * v for n, v in lst) / sum(n for n, _ in lst))
             for r, lst in sorted(event.items())}
    cohort_att = {gg: float(sum(n * v for n, v in lst)
                            / sum(n for n, _ in lst))
                  for gg, lst in sorted(cohort_att.items())}

    # equivalence with the imputation estimator (no covariates)
    match = None
    if X is None:
        from .boryis import impute_untreated

        Y0, _, _, _ = impute_untreated(Y, treated)
        tau = Y - Y0
        match = float(max(abs(att[(gg, t)] - tau[(g == gg), t].mean())
                          for gg, t in cells))

    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - z * se, est + z * se),
            "att_gt": att,
            "event": event,
            "cohort_att": cohort_att,
            "cell_sizes": n_cell,
            "min_cell_size": int(min(n_cell.values())),
            "n_interactions": len(cells),
            "n_parameters": int(Z.shape[1]),
            "n_observations": int(n_obs),
            "covariate_coef": (coef[k + len(cells):] if X is not None else None),
            "matches_imputation": match,
            "equivalence_note": (
                "with no covariates this estimator and the "
                "Borusyak-Jaravel-Spiess imputation estimator are the same "
                "number; matches_imputation is the measured discrepancy"
            ),
            "saturation_note": (
                "one parameter per treated cohort-period cell means no "
                "pooling and no negative weights, at the cost of thin cells"
            ),
            "se_note": "clustered on unit",
            "method": "Wooldridge (2021) extended two-way fixed effects",
        }
    )


def cheatsheet():
    return (
        "wbcide: saturated cohort-by-period TWFE; the interaction "
        "coefficients ARE the ATT(g,t), equal to the imputation estimator"
    )
