# morie.fn -- function file (rootcoder007/morie)
"""Doubly robust (AIPW) difference-in-differences with covariates."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["aipw_did"]


def _logit_fit(X, y, max_iter=100, tol=1e-9):
    """Newton-Raphson logistic regression; returns the fitted probabilities."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(D.shape[1])
    for _ in range(max_iter):
        eta = np.clip(D @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10)
        grad = D.T @ (y - p)
        H = (D * W[:, None]).T @ D
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(D @ beta, -35, 35)
    return 1.0 / (1.0 + np.exp(-eta))


def _ols_predict(X, y, fit_rows):
    """Fit y ~ 1 + X on ``fit_rows`` only, predict for everyone."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    b, *_ = np.linalg.lstsq(D[fit_rows], y[fit_rows], rcond=None)
    return D @ b


def aipw_did(y_pre, y_post, D, X, trim=0.995, alpha=0.05):
    r"""Doubly robust difference-in-differences (Sant'Anna & Zhao 2020).

    For panel data the outcome change :math:`\Delta Y = Y_{post} - Y_{pre}`
    carries the unit fixed effect away, and the ATT is estimated by
    combining a propensity score with an outcome regression:

    .. math::

        \widehat{ATT} = \frac{1}{E_n[D]}\,E_n\!\left[
            \left(D - \frac{(1-D)\,\hat\pi(X)}{1-\hat\pi(X)}\right)
            \bigl(\Delta Y - \hat m_0(X)\bigr)\right]

    with :math:`\hat\pi(X)` the probability of treatment and
    :math:`\hat m_0(X)` the regression of :math:`\Delta Y` on covariates
    fitted **on the comparison group only**.

    "Doubly robust" is the point: the estimator stays consistent if
    *either* the propensity score or the outcome regression is correctly
    specified, not necessarily both. That is why the two nuisance pieces
    are worth fitting even though either alone would give an estimator.
    A test here checks that property directly, by deliberately
    misspecifying one at a time.

    Parallel trends is still assumed, now conditional on X. No amount of
    double robustness rescues the estimate if trends differ between
    groups in a way X does not capture.

    Extreme propensity scores are the practical failure mode: the weight
    :math:`\hat\pi/(1-\hat\pi)` explodes as :math:`\hat\pi \to 1`, so a
    handful of comparison units can dominate the estimate. Scores are
    trimmed at ``trim`` and the number trimmed is reported rather than
    hidden.

    Parameters
    ----------
    y_pre, y_post : array-like, shape (n,)
        Outcome before and after, same units in the same order.
    D : array-like, shape (n,)
        Binary treatment indicator.
    X : array-like, shape (n, k)
        Covariates. A one-dimensional input is read as a single column.
    trim : float, default 0.995
        Upper bound on the fitted propensity score.
    alpha : float, default 0.05
        Significance level for the interval.

    Returns
    -------
    RichResult
        keys: ``att``, ``estimate``, ``se``, ``ci_low``, ``ci_high``,
        ``statistic``, ``p_value``, ``n_treated``, ``n_control``,
        ``n_trimmed``, ``ps_min``, ``ps_max``, ``method``.

    References
    ----------
    Sant'Anna, P. H. C. & Zhao, J. (2020). Doubly robust
    difference-in-differences estimators. *Journal of Econometrics*,
    219(1), 101-122.
    """
    pre = np.asarray(y_pre, dtype=float).ravel()
    post = np.asarray(y_post, dtype=float).ravel()
    d = np.asarray(D, dtype=float).ravel()
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    n = pre.size
    if not (post.size == n and d.size == n and Xa.shape[0] == n):
        raise ValueError(f"y_pre, y_post, D and X must share a length; got {n}, {post.size}, {d.size}, {Xa.shape[0]}.")
    if not set(np.unique(d)) <= {0.0, 1.0}:
        raise ValueError("D must be binary (0/1).")
    if d.sum() < 2 or (1 - d).sum() < 2:
        raise ValueError(f"Need at least 2 units per arm; got {int(d.sum())} treated, {int((1 - d).sum())} control.")
    if not all(np.all(np.isfinite(v)) for v in (pre, post, Xa)):
        raise ValueError("y_pre, y_post and X must be finite.")
    if not 0 < trim < 1:
        raise ValueError(f"trim must lie in (0, 1), got {trim}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")

    dy = post - pre
    ps = _logit_fit(Xa, d)
    n_trim = int(np.sum(ps > trim))
    ps = np.clip(ps, 1e-6, trim)
    m0 = _ols_predict(Xa, dy, d == 0)

    resid = dy - m0
    w1 = d
    w0 = (1 - d) * ps / (1 - ps)
    # Hajek form: each weight normalised by its own mean, which keeps the
    # estimator invariant to a rescaling of the weights.
    t1 = np.mean(w1 * resid) / np.mean(w1)
    t0 = np.mean(w0 * resid) / np.mean(w0)
    att = float(t1 - t0)

    # Influence function, so the standard error accounts for the
    # weighting rather than treating the weights as fixed.
    inf = (w1 * (resid - t1)) / np.mean(w1) - (w0 * (resid - t0)) / np.mean(w0)
    se = float(np.std(inf, ddof=1) / np.sqrt(n))

    if se > 0:
        z = att / se
        crit = stats.norm.ppf(1 - alpha / 2)
        lo, hi = att - crit * se, att + crit * se
        p = float(2 * stats.norm.sf(abs(z)))
    else:
        z, lo, hi, p = np.nan, att, att, np.nan

    return RichResult(
        title="Doubly robust difference-in-differences",
        payload={
            "att": att,
            "estimate": att,
            "se": se,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "statistic": float(z),
            "p_value": p,
            "n_treated": int(d.sum()),
            "n_control": int((1 - d).sum()),
            "n_trimmed": n_trim,
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "method": "AIPW / doubly robust DiD (Sant'Anna & Zhao 2020), panel",
        },
    )


def cheatsheet():
    return "aiptdd: doubly robust (AIPW) difference-in-differences"
