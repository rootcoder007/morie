# morie.fn -- function file (rootcoder007/morie)
"""Shrinkage propensity model via a Bayesian (ridge) prior."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["shrinkage_propensity"]


def shrinkage_propensity(A, H, prior_sd=1.0, max_iter=100, tol=1e-9):
    r"""Penalised logistic propensity with a normal prior on the coefficients.

    Maximises the log posterior

    .. math:: \ell(\beta) - \frac{1}{2\tau^2}\|\beta_{-0}\|^2,

    i.e. a normal :math:`N(0, \tau^2)` prior on every slope (the
    intercept is left unpenalised). With many covariates or near-
    separation the unpenalised MLE diverges and the fitted propensities
    pile up at 0 and 1, which makes IP weights explode; shrinkage keeps
    them inside the interior. Smaller ``prior_sd`` = more shrinkage.

    Parameters
    ----------
    A : array-like of {0, 1}, shape (n,)
        Treatment.
    H : array-like, shape (n, p) or (n,)
        Covariates (standardised internally so the prior is scale-free).
    prior_sd : float, default 1.0
        Prior standard deviation tau on the standardised scale.
    max_iter, tol :
        Newton-Raphson controls.

    Returns
    -------
    RichResult
        keys: ``propensity`` (n,), ``coefficients`` (on the
        standardised scale), ``ps_min``, ``ps_max``, ``prior_sd``,
        ``n``, ``method``.

    References
    ----------
    Gelman, A., Jakulin, A., Pittau, M. G. & Su, Y.-S. (2008). A
    weakly informative default prior distribution for logistic and
    other regression models. *The Annals of Applied Statistics*, 2(4),
    1360-1383.
    """
    A = np.asarray(A, dtype=float).ravel()
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H[:, None]
    n, p = H.shape
    if A.size != n:
        raise ValueError(f"H has {n} rows but A has {A.size}.")
    tau = float(prior_sd)
    if tau <= 0:
        raise ValueError(f"prior_sd must be positive, got {tau}.")

    sd = H.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Hs = (H - H.mean(axis=0)) / sd
    D = np.column_stack([np.ones(n), Hs])
    pen = np.ones(p + 1) / tau**2
    pen[0] = 0.0  # intercept unpenalised

    beta = np.zeros(p + 1)
    for _ in range(int(max_iter)):
        eta = np.clip(D @ beta, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        W = np.maximum(mu * (1 - mu), 1e-10)
        grad = D.T @ (A - mu) - pen * beta
        Hess = (D * W[:, None]).T @ D + np.diag(pen)
        try:
            step = np.linalg.solve(Hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(Hess) @ grad
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    ps = 1 / (1 + np.exp(-np.clip(D @ beta, -35, 35)))
    return RichResult(
        payload={
            "propensity": ps,
            "coefficients": beta,
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "prior_sd": tau,
            "n": int(n),
            "method": "Shrinkage (normal-prior penalised) logistic propensity",
        }
    )


def cheatsheet():
    return "shrtgr: MAP logistic with N(0, tau^2) slope prior; keeps e(X) off 0/1"
