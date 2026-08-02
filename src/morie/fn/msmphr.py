# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural Cox model fitted with IPTW."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit
from .survmd import _cox_newton

__all__ = ["msm_proportional_hazards"]


def _weighted_cox(X, time, event, w, max_iter=100, tol=1e-9):
    """Breslow partial likelihood with case weights."""
    order = np.argsort(time)
    X, time, event, w = X[order], time[order], event[order], w[order]
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        ew = w * np.exp(eta)
        S0 = np.cumsum(ew[::-1])[::-1]
        S1 = np.cumsum((ew[:, None] * X)[::-1], axis=0)[::-1]
        grad = np.zeros(p)
        H = np.zeros((p, p))
        for i in np.flatnonzero(event == 1):
            xbar = S1[i] / S0[i]
            grad += w[i] * (X[i] - xbar)
            S2 = ((ew[i:, None] * X[i:]).T @ X[i:]) / S0[i]
            H += w[i] * (S2 - np.outer(xbar, xbar))
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def msm_proportional_hazards(time, event, treatment, covariates):
    r"""Cox MSM: IPTW-weighted proportional hazards on treatment alone.

    Fits stabilised inverse-probability-of-treatment weights from a
    logistic model of treatment given the confounders, then a Cox model
    containing *only* treatment,

    .. math:: \lambda(t \mid \bar a) = \lambda_0(t) e^{\beta a},

    so :math:`e^\beta` is the marginal (causal) hazard ratio rather
    than a hazard ratio conditional on the confounders. The
    unweighted fit is returned alongside: when the two differ, the
    confounding was doing work.

    Parameters
    ----------
    time : array-like, shape (n,)
        Follow-up times (positive).
    event : array-like of {0, 1}, shape (n,)
        Event indicator.
    treatment : array-like of {0, 1}, shape (n,)
        Baseline treatment.
    covariates : array-like, shape (n, p) or (n,)
        Confounders for the weight model.

    Returns
    -------
    RichResult
        keys: ``log_hr``, ``hazard_ratio``, ``log_hr_unweighted``,
        ``weights``, ``ess``, ``n_events``, ``n``, ``method``.

    References
    ----------
    Hernan, M. A., Brumback, B. & Robins, J. M. (2000). Marginal
    structural models to estimate the causal effect of zidovudine on
    the survival of HIV-positive men. *Epidemiology*, 11(5), 561-570.
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    A = np.asarray(treatment, dtype=float).ravel()
    C = np.asarray(covariates, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    n = time.size
    if not (event.size == n and A.size == n and C.shape[0] == n):
        raise ValueError("time, event, treatment, covariates must share their first dimension.")
    if np.any(time <= 0):
        raise ValueError("time must be positive.")
    for v, name in ((event, "event"), (A, "treatment")):
        if not np.all(np.isin(v, (0.0, 1.0))):
            raise ValueError(f"{name} must be binary 0/1.")
    if event.sum() < 5:
        raise ValueError(f"need at least 5 events, got {int(event.sum())}.")
    if A.sum() == 0 or A.sum() == n:
        raise ValueError("need both treatment arms.")

    e = np.clip(_logit_fit(C, A), 0.01, 0.99)
    p_marg = A.mean()
    num = np.where(A == 1, p_marg, 1 - p_marg)
    den = np.where(A == 1, e, 1 - e)
    sw = num / den

    beta = _weighted_cox(A[:, None], time, event, sw)
    beta_u = _cox_newton(A[:, None], time, event)

    return RichResult(
        payload={
            "log_hr": float(beta[0]),
            "hazard_ratio": float(np.exp(beta[0])),
            "log_hr_unweighted": float(beta_u[0]),
            "weights": sw,
            "ess": float(sw.sum() ** 2 / (sw**2).sum()),
            "n_events": int(event.sum()),
            "n": int(n),
            "method": "Marginal structural Cox model (stabilised IPTW, treatment-only)",
        }
    )


def cheatsheet():
    return "msmphr: stabilised IPTW then a Cox model with treatment only = marginal HR"
