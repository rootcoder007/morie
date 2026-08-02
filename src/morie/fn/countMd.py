# morie.fn -- function file (rootcoder007/morie)
"""Mediation for a count outcome."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["count_mediation"]


def _poisson_irls(X, y, max_iter=100, tol=1e-10):
    """Poisson GLM with a log link by iteratively reweighted least squares."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    b = np.zeros(D.shape[1])
    b[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(max_iter):
        eta = np.clip(D @ b, -30, 30)
        mu = np.exp(eta)
        W = np.maximum(mu, 1e-10)
        z = eta + (y - mu) / W
        A = (D * W[:, None]).T @ D
        rhs = (D * W[:, None]).T @ z
        try:
            new = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            new = np.linalg.pinv(A) @ rhs
        if np.max(np.abs(new - b)) < tol:
            b = new
            break
        b = new
    return b


def count_mediation(y, x, m, c=None):
    r"""Mediation with a Poisson (log-link) outcome model.

    The mediator model stays linear; the count outcome is fitted by a
    Poisson GLM,

    .. math:: \log E[Y \mid X, M, C]
              = \theta_0 + \theta_1 X + \theta_2 M + \theta_3' C.

    On the rate-ratio scale VanderWeele's rare-outcome approximation
    gives multiplicative effects

    .. math:: \mathrm{RR}^{NDE} = e^{\theta_1}, \qquad
              \mathrm{RR}^{NIE} = e^{\theta_2 \beta_1},

    whose product is the total rate ratio -- effects *multiply* here
    rather than adding, which is the substantive difference from the
    linear case.

    Parameters
    ----------
    y : array-like of nonnegative ints, shape (n,)
        Count outcome.
    x, m : array-like, shape (n,)
        Exposure and mediator.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``rr_nde``, ``rr_nie``, ``rr_total``,
        ``log_nde``, ``log_nie``, ``coefficients``, ``n``, ``method``.

    References
    ----------
    VanderWeele, T. J. (2015). *Explanation in Causal Inference*.
    Oxford University Press. Ch. 2 (mediation with non-continuous
    outcomes; effects on the ratio scale).
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    n = y.size
    if not (x.size == n and m.size == n):
        raise ValueError("y, x, m must have equal length.")
    if np.any(y < 0) or np.any(y != np.floor(y)):
        raise ValueError("y must be nonnegative counts.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but y has {n}.")
    if n < C.shape[1] + 6:
        raise ValueError("too few observations for the mediator and outcome models.")

    bm, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x, C]), m, rcond=None)
    beta1 = float(bm[1])
    th = _poisson_irls(np.column_stack([x, m, C]), y)
    t1, t2 = float(th[1]), float(th[2])

    log_nde, log_nie = t1, t2 * beta1
    return RichResult(
        payload={
            "rr_nde": float(np.exp(log_nde)),
            "rr_nie": float(np.exp(log_nie)),
            "rr_total": float(np.exp(log_nde + log_nie)),
            "log_nde": log_nde,
            "log_nie": float(log_nie),
            "coefficients": {"beta1": beta1, "theta1": t1, "theta2": t2},
            "n": int(n),
            "method": "Count-outcome mediation (Poisson log link, rate-ratio effects)",
        }
    )


def cheatsheet():
    return "countMd: RR_NDE = exp(theta1), RR_NIE = exp(theta2*beta1), product = total RR"
