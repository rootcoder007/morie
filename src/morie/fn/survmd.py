# morie.fn -- function file (rootcoder007/morie)
"""Mediation with a survival outcome."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survival_mediation"]


def _cox_newton(X, time, event, max_iter=100, tol=1e-9):
    """Breslow partial-likelihood Cox fit by Newton-Raphson."""
    order = np.argsort(time)
    X, time, event = X[order], time[order], event[order]
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        w = np.exp(eta)
        # risk set n..i (times sorted ascending): reverse cumulative sums
        S0 = np.cumsum(w[::-1])[::-1]
        S1 = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]
        grad = np.zeros(p)
        H = np.zeros((p, p))
        for i in np.flatnonzero(event == 1):
            xbar = S1[i] / S0[i]
            grad += X[i] - xbar
            S2 = ((w[i:, None] * X[i:]).T @ X[i:]) / S0[i]
            H += S2 - np.outer(xbar, xbar)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def survival_mediation(time, event, x, m, c=None):
    r"""Mediation with a time-to-event outcome (Cox outcome model).

    The mediator model is linear; the outcome is a Cox proportional
    hazards model

    .. math:: \lambda(t \mid X, M, C)
              = \lambda_0(t)\, e^{\theta_1 X + \theta_2 M + \theta_3' C}.

    Under the rare-outcome approximation the natural effects are
    hazard ratios

    .. math:: \mathrm{HR}^{NDE} = e^{\theta_1}, \qquad
              \mathrm{HR}^{NIE} = e^{\theta_2 \beta_1},

    with the total hazard ratio their product. The approximation is
    the price of the semiparametric baseline hazard; VanderWeele
    states it explicitly and so does this docstring.

    Parameters
    ----------
    time : array-like, shape (n,)
        Follow-up times (positive).
    event : array-like of {0, 1}, shape (n,)
        1 = event observed, 0 = right-censored.
    x, m : array-like, shape (n,)
        Exposure and mediator.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``hr_nde``, ``hr_nie``, ``hr_total``, ``log_nde``,
        ``log_nie``, ``coefficients``, ``n_events``, ``n``, ``method``.

    References
    ----------
    VanderWeele, T. J. (2011). Causal mediation analysis with survival
    data. *Epidemiology*, 22(4), 582-585.
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    n = time.size
    if not (event.size == n and x.size == n and m.size == n):
        raise ValueError("time, event, x, m must have equal length.")
    if np.any(time <= 0):
        raise ValueError("time must be positive.")
    if not np.all(np.isin(event, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    if event.sum() < 5:
        raise ValueError(f"need at least 5 events, got {int(event.sum())}.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but time has {n}.")

    bm, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x, C]), m, rcond=None)
    beta1 = float(bm[1])
    th = _cox_newton(np.column_stack([x, m, C]), time, event)
    t1, t2 = float(th[0]), float(th[1])

    log_nde, log_nie = t1, t2 * beta1
    return RichResult(
        payload={
            "hr_nde": float(np.exp(log_nde)),
            "hr_nie": float(np.exp(log_nie)),
            "hr_total": float(np.exp(log_nde + log_nie)),
            "log_nde": log_nde,
            "log_nie": float(log_nie),
            "coefficients": {"beta1": beta1, "theta1": t1, "theta2": t2},
            "n_events": int(event.sum()),
            "n": int(n),
            "method": "Survival mediation (Cox outcome, rare-outcome hazard-ratio effects)",
        }
    )


def cheatsheet():
    return "survmd: HR_NDE = exp(theta1), HR_NIE = exp(theta2*beta1) from a Cox outcome model"
