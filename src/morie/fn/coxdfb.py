# morie.fn -- function file (rootcoder007/morie)
"""DFBETA influence measures for a Cox model."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["cox_dfbeta_influence"]


def cox_dfbeta_influence(fit):
    r"""Approximate change in each coefficient from deleting each subject.

    The score residual for subject :math:`i` scaled by the inverse
    information,

    .. math::
        \mathrm{dfbeta}_i \approx \mathcal{I}(\hat\beta)^{-1} \, L_i,

    where :math:`L_i` is subject :math:`i`'s score contribution. This is the
    one-step approximation to refitting without that subject, and it is
    accurate enough to rank influence without paying :math:`n` refits.

    Influence is not outlyingness. A subject can have an extreme covariate and
    almost no influence (if they are censored early, before contributing to
    many risk sets), or an unremarkable covariate and large influence (if they
    fail early while still in everyone's risk set). Reporting dfbeta scaled by
    the standard error -- ``dfbetas`` here -- puts the change on the same
    footing as the uncertainty, which is the comparison that matters: a value
    near 1 means one subject moves the estimate by a full standard error.

    Parameters
    ----------
    fit : mapping
        A result from one of the Cox fitters.

    Returns
    -------
    RichResult
        ``dfbeta`` ``(n, p)``, ``dfbetas`` (SE-scaled), ``max_influence``,
        ``most_influential``.

    References
    ----------
    Cain, K. C., & Lange, N. T. (1984). Approximate case influence for the
        proportional hazards regression model. *Biometrics*, 40(2), 493-499.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    Shapes line up with the data, one row per subject.

    >>> import numpy as np
    >>> from morie.fn.efrnt import efron_tie_correction
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> T = rng.exponential(1 / np.exp(X @ [0.8, -0.5]))
    >>> C = rng.exponential(2.0, 300)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_dfbeta_influence(efron_tie_correction(t, e, X))
    >>> r["dfbeta"].shape
    (300, 2)

    In well-behaved data no single subject moves a coefficient by a whole
    standard error.

    >>> bool(np.max(np.abs(r["dfbetas"])) < 1.0)
    True

    Influence is not outlyingness, and the two implanted subjects show it.
    A subject with an extreme covariate who fails immediately is exactly what
    the model predicts, so it barely moves the estimate.

    >>> agrees = cox_dfbeta_influence(efron_tie_correction(
    ...     np.r_[t, 1e-4], np.r_[e, 1.0], np.vstack([X, [[6.0, 0.0]]])))
    >>> bool(agrees["most_influential"] != 300)
    True

    The same covariate on a subject who instead survives longest contradicts
    the model, and moves a coefficient by nearly three standard errors.

    >>> denies = cox_dfbeta_influence(efron_tie_correction(
    ...     np.r_[t, t.max() * 1.5], np.r_[e, 1.0], np.vstack([X, [[6.0, 0.0]]])))
    >>> int(denies["most_influential"])
    300
    >>> bool(denies["max_influence"] > 2.0)
    True
    """
    t = np.asarray(fit["time"], dtype=float)
    e = np.asarray(fit["event"], dtype=float)
    X = np.atleast_2d(np.asarray(fit["X"], dtype=float))
    beta = np.asarray(fit["beta"], dtype=float).ravel()
    I = np.asarray(fit["information"], dtype=float)
    se = np.asarray(fit["se"], dtype=float).ravel()
    n, p = X.shape
    w = np.exp(np.clip(X @ beta, -500, 500))

    # Score residuals: subject i's contribution to the score equation.
    L = np.zeros((n, p))
    for ut in np.unique(t[e == 1]):
        at_risk = t >= ut
        died = at_risk & (t == ut) & (e == 1)
        wr = w[at_risk]
        S0 = wr.sum()
        mu = (wr @ X[at_risk]) / S0
        d = int(died.sum())
        L[died] += X[died] - mu
        L[at_risk] -= d * (w[at_risk][:, None] * (X[at_risk] - mu)) / S0

    try:
        Iinv = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        Iinv = np.linalg.pinv(I)
    dfbeta = L @ Iinv
    with np.errstate(divide="ignore", invalid="ignore"):
        dfbetas = dfbeta / np.where(se > 0, se, np.nan)
    worst = int(np.argmax(np.abs(dfbetas).max(axis=1)))
    return RichResult(
        title="Cox DFBETA influence",
        summary_lines=[("n", n), ("max |dfbetas|", float(np.nanmax(np.abs(dfbetas)))),
                       ("most influential", worst)],
        payload={
            "dfbeta": dfbeta, "dfbetas": dfbetas, "score_residuals": L,
            "max_influence": float(np.nanmax(np.abs(dfbetas))),
            "most_influential": worst, "method": "cox_dfbeta_influence",
        },
    )


def cheatsheet():
    return "coxdfb: influence != outlyingness; read dfbetas (SE-scaled), where ~1 means one subject moves beta by 1 SE"
