# morie.fn -- function file (rootcoder007/morie)
"""Iteratively reweighted least squares -- ESL Sec 4.4.1."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_iwls"]


def esl_iwls(X, y, beta0=None, family="binomial", max_iter=50, tol=1e-8, add_intercept=True):
    r"""Fit a GLM by iteratively reweighted least squares (Fisher scoring).

    Each iteration solves a weighted least-squares problem in the adjusted
    response :math:`z`:

    .. math::
        \beta^{new} = (X^\top W X)^{-1} X^\top W z, \qquad
        z = X\beta + W^{-1}(y - \mu),

    with :math:`W = \operatorname{diag}(\mu_i(1-\mu_i))` for the binomial
    family and :math:`\operatorname{diag}(\mu_i)` for Poisson. For canonical
    links Fisher scoring and Newton-Raphson coincide, which is why this
    single loop serves both.

    Separation is the failure mode worth naming: when a linear combination
    perfectly splits the classes the MLE runs off to infinity, the weights go
    to zero, and the coefficients diverge while the likelihood keeps
    improving. That is detected and reported rather than returned as a fit.

    Parameters
    ----------
    X : array-like
        Design matrix, shape ``(n, p)``, without an intercept column unless
        ``add_intercept=False``.
    y : array-like
        Response. 0/1 for ``"binomial"``, non-negative counts for
        ``"poisson"``.
    beta0 : array-like, optional
        Starting coefficients. Defaults to zeros.
    family : {"binomial", "poisson"}
        Error family with its canonical link.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance on the coefficient change.
    add_intercept : bool
        Prepend a column of ones.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``z``, ``p_value``, ``fitted``, ``loglik``,
        ``deviance``, ``n_iter``, ``converged``, ``separated``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Coefficient recovery on data simulated from the model.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(4000, 2))
    >>> eta = -0.5 + 1.5 * X[:, 0] - 1.0 * X[:, 1]
    >>> y = (rng.random(4000) < 1 / (1 + np.exp(-eta))).astype(float)
    >>> b = esl_iwls(X, y)["beta"]
    >>> [bool(abs(b[i] - t) < 0.15) for i, t in enumerate([-0.5, 1.5, -1.0])]
    [True, True, True]

    Perfect separation is flagged instead of silently returning huge
    coefficients.

    >>> r = esl_iwls([[-2.0], [-1.0], [1.0], [2.0]], [0.0, 0.0, 1.0, 1.0])
    >>> bool(r["separated"])
    True

    >>> esl_iwls([[1.0], [2.0]], [0.0, 2.0])
    Traceback (most recent call last):
        ...
    ValueError: binomial y must be 0/1
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size}")
    if family == "binomial" and not np.all((y == 0) | (y == 1)):
        raise ValueError("binomial y must be 0/1")
    if family == "poisson" and np.any(y < 0):
        raise ValueError("poisson y must be non-negative")
    if family not in ("binomial", "poisson"):
        raise ValueError('family must be "binomial" or "poisson"')
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else np.asarray(beta0, dtype=float).ravel()
    if beta.size != p:
        raise ValueError(f"beta0 must have {p} entries")

    converged = False
    for it in range(1, max_iter + 1):
        eta = X @ beta
        if family == "binomial":
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
            w = np.clip(mu * (1 - mu), 1e-10, None)
        else:
            mu = np.exp(np.clip(eta, -500, 500))
            w = np.clip(mu, 1e-10, None)
        z = eta + (y - mu) / w
        WX = X * w[:, None]
        XtWX = X.T @ WX
        try:
            new = np.linalg.solve(XtWX, WX.T @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(XtWX, WX.T @ z, rcond=None)[0]
        delta = float(np.max(np.abs(new - beta)))
        beta = new
        if delta < tol:
            converged = True
            break

    eta = X @ beta
    if family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        ll = float(np.sum(y * np.log(mu + 1e-300) + (1 - y) * np.log(1 - mu + 1e-300)))
    else:
        mu = np.exp(np.clip(eta, -500, 500))
        w = np.clip(mu, 1e-10, None)
        from scipy.special import gammaln

        ll = float(np.sum(y * np.log(mu + 1e-300) - mu - gammaln(y + 1)))

    # Under separation the fitted probabilities are driven to exactly 0/1 while
    # the coefficients are still climbing, so the probabilities detect it a long
    # way before any coefficient threshold does.
    separated = bool(
        family == "binomial"
        and (np.max(np.abs(beta)) > 25 or np.all(np.abs(mu - y) < 1e-6))
    )
    try:
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = beta / se
    from scipy.stats import norm

    warn = []
    if not converged:
        warn.append(f"IRLS did not converge in {max_iter} iterations")
    if separated:
        warn.append(
            "the fit shows (quasi-)complete separation -- fitted probabilities pinned "
            "at 0/1 and/or coefficients past |25|. The MLE does not exist and these "
            "estimates are not interpretable"
        )
    return RichResult(
        title=f"IRLS ({family})",
        summary_lines=[("n", n), ("p", p), ("loglik", ll), ("iterations", it)],
        warnings=warn,
        payload={
            "beta": beta, "se": se, "z": zstat,
            "p_value": 2 * norm.sf(np.abs(zstat)),
            "fitted": mu, "loglik": ll, "deviance": float(-2 * ll),
            "n_iter": int(it), "converged": bool(converged),
            "separated": separated, "family": family,
            "method": "esl_iwls",
        },
    )


def cheatsheet():
    return "esliwls: Fisher scoring for binomial/Poisson GLMs; flags separation instead of returning huge betas"
