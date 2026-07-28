# morie.fn -- function file (rootcoder007/morie)
"""Maximum likelihood as an M-estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_mle"]


def wasserman_mle(data, f, theta0, bounds=None, se=True):
    r"""Maximum likelihood estimate
    :math:`\hat\theta = \arg\max_\theta \ell(\theta)`, with
    :math:`\ell(\theta) = \sum_i \log f(X_i;\theta)`.

    Maximum likelihood is the M-estimator with criterion
    :math:`m_\theta(x) = \log f(x;\theta)`: it maximises an empirical
    average :math:`\mathbb P_n m_\theta`, and the general theory of
    Kosorok's Ch. 14 applies to it directly. That framing is what
    supplies the conditions for consistency and asymptotic normality
    -- identifiability plus a well-separated maximum, not merely a
    stationary point of the derivative.

    The standard error comes from the OBSERVED information,
    :math:`-\partial^2\ell/\partial\theta^2` evaluated at
    :math:`\hat\theta` and computed by central differences, rather
    than the expected (Fisher) information, which would require an
    expectation this function has no way to take. The two agree
    asymptotically and the observed one is what is available.

    Two guards matter more than the optimisation itself. If the
    Hessian is not negative definite the point found is not a
    maximum, and the reported standard error would be the square root
    of a negative number; ``se`` is then ``None`` and
    ``is_maximum`` is ``False`` rather than the result being quietly
    returned as if it were fine. And a likelihood that is unbounded
    -- a normal mixture with a free component variance, say -- has no
    maximum at all, so a finite answer here means the optimiser
    stopped somewhere, not that the estimate exists.

    Parameters
    ----------
    data : array-like
        Sample.
    f : callable
        ``f(x, theta)`` returning the DENSITY (not the log density)
        at each observation.
    theta0 : array-like
        Starting value; its length sets the dimension.
    bounds : sequence of pairs, optional
        Box constraints passed to the optimiser.
    se : bool, default True
        Compute the observed-information standard error.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``se``, ``loglik``, ``observed_information``,
        ``is_maximum``, ``converged``, ``n_params``, ``n``,
        ``information_used``, ``method``.

    References
    ----------
    Kosorok, M. R. (2008), *Introduction to Empirical Processes and
    Semiparametric Inference*, Springer, Ch. 14 (M-estimators) and
    Sec. 2.2.6. Fisher (1922) for maximum likelihood itself.
    """
    from scipy import optimize

    d = np.asarray(data, dtype=float)
    n = d.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    t0 = np.atleast_1d(np.asarray(theta0, dtype=float)).ravel()

    def negll(th):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.asarray(f(d, th), dtype=float).ravel()
            if np.any(v <= 0) or not np.all(np.isfinite(v)):
                return np.inf
            return -float(np.sum(np.log(v)))

    if not np.isfinite(negll(t0)):
        raise ValueError(
            "the log-likelihood is not finite at theta0; the density is zero "
            "or negative at some observation there.")
    r = optimize.minimize(negll, t0, method="Nelder-Mead", bounds=bounds,
                          options={"maxiter": 20000, "fatol": 1e-12,
                                   "xatol": 1e-10})
    th = np.asarray(r.x, dtype=float).ravel()
    ll = -float(r.fun)
    out = {"estimate": th if th.size > 1 else float(th[0]),
           "loglik": ll, "converged": bool(r.success),
           "n_params": int(th.size), "n": int(n),
           "information_used": "observed, -d2 loglik/dtheta2 at theta_hat, by "
                               "central differences; the expected (Fisher) "
                               "information would need an expectation this "
                               "function cannot take",
           "m_estimator_note": "MLE is the M-estimator with criterion "
                               "log f(x; theta); consistency needs a "
                               "WELL-SEPARATED maximum, not just a stationary "
                               "point",
           "method": "Maximum likelihood as an M-estimator (Kosorok Ch. 14)"}
    if not se:
        out["se"] = None
        out["is_maximum"] = None
        return RichResult(payload=out)

    k = th.size
    step = np.maximum(1e-5, 1e-4 * np.abs(th))
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ei[i] = step[i]
            ej = np.zeros(k); ej[j] = step[j]
            H[i, j] = ((negll(th + ei + ej) - negll(th + ei - ej)
                        - negll(th - ei + ej) + negll(th - ei - ej))
                       / (4 * step[i] * step[j]))
    H = (H + H.T) / 2.0                       # Hessian of the NEGATIVE loglik
    eig = np.linalg.eigvalsh(H)
    is_max = bool(np.all(eig > 0) and np.all(np.isfinite(eig)))
    out["observed_information"] = H
    out["is_maximum"] = is_max
    if is_max:
        cov = np.linalg.inv(H)
        s = np.sqrt(np.diag(cov))
        out["se"] = s if k > 1 else float(s[0])
        out["covariance"] = cov
    else:
        out["se"] = None
        out["not_a_maximum_note"] = (
            "the observed information is not positive definite, so the point "
            "found is not a maximum and no standard error is reported")
    return RichResult(payload=out)


def cheatsheet():
    return "wsmmle: MLE is an M-estimator; a non-positive-definite Hessian means it is not a maximum"
