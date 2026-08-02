# morie.fn -- function file (rootcoder007/morie)
"""Causal mediation for a binary outcome by inverse odds-ratio weighting."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["binary_outcome_mediation"]


def _logit(X, y, max_iter=200, tol=1e-10):
    """Newton-Raphson logistic regression; returns coefficients."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(D.shape[1])
    for _ in range(max_iter):
        eta = np.clip(D @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10)
        H = (D * W[:, None]).T @ D
        g = D.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _wlogit(X, y, w, max_iter=200, tol=1e-10):
    """Weighted logistic regression; returns coefficients."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(D.shape[1])
    for _ in range(max_iter):
        eta = np.clip(D @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10) * w
        H = (D * W[:, None]).T @ D
        g = D.T @ (w * (y - p))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def binary_outcome_mediation(X, M, Y, C=None, B=0, seed=None, alpha=0.05):
    r"""Natural direct and indirect effects for a binary outcome.

    Uses Tchetgen Tchetgen's inverse odds-ratio weighting. Fit the
    exposure model :math:`P(X = 1 \mid M, C)` and form, for the exposed,

    .. math:: w_i = \exp\!\left(-\hat\gamma' M_i\right)

    the reciprocal of the exposure-mediator odds ratio. Refitting the
    total-effect outcome model on those weights removes the mediated
    part, so the weighted fit gives the **natural direct effect** and
    the difference from the unweighted total gives the **natural
    indirect effect**, both on the log-odds scale.

    Two reasons this exists rather than reusing the linear machinery.
    It needs no model for the mediator's distribution, which the
    product-of-coefficients route does. And logistic coefficients are
    not comparable across models -- each is scaled by its own residual
    variance -- so the familiar :math:`ab` and :math:`c - c'`
    decompositions simply do not hold on the logit scale. That
    non-collapsibility is why :func:`morie.fn.bkmed.baron_kenny` should
    not be run on a binary outcome and called mediation.

    Effects come back as odds ratios as well as log-odds. Identification
    rests on the usual four assumptions -- no unmeasured exposure-outcome,
    mediator-outcome or exposure-mediator confounding, and no
    mediator-outcome confounder affected by the exposure. None is
    checkable here.

    Parameters
    ----------
    X : array-like, shape (n,)
        Binary exposure.
    M : array-like, shape (n,) or (n, p)
        Mediator or mediators.
    Y : array-like, shape (n,)
        Binary outcome.
    C : array-like, optional
        Baseline covariates, adjusted for in every model.
    B : int, default 0
        Bootstrap replicates for standard errors. The weights are
        themselves estimated, so a model-based standard error from the
        weighted fit understates the uncertainty; with ``B = 0`` no
        standard error is reported rather than a wrong one.
    seed : int, optional
        Seed for the bootstrap.
    alpha : float, default 0.05
        Significance level for the bootstrap interval.

    Returns
    -------
    RichResult
        keys: ``total``, ``direct``, ``indirect`` (log-odds),
        ``or_total``, ``or_direct``, ``or_indirect``,
        ``proportion_mediated``, ``se``, ``ci_low``, ``ci_high``,
        ``p_value``, ``n``, ``B``, ``method``.

    References
    ----------
    Tchetgen Tchetgen, E. J. (2013). Inverse odds ratio-weighted
    estimation for causal mediation analysis. *Statistics in Medicine*,
    32(26), 4567-4580.
    """
    x = np.asarray(X, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    m = np.asarray(M, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    n = x.size
    if not (y.size == n and m.shape[0] == n):
        raise ValueError(f"X, M and Y must share a length; got {n}, {m.shape[0]}, {y.size}.")
    if not set(np.unique(x)) <= {0.0, 1.0}:
        raise ValueError("X must be binary (0/1).")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("Y must be binary (0/1); this estimator is for a binary outcome.")
    cov = None if C is None else np.asarray(C, dtype=float).reshape(n, -1)
    for arr, nm in ((x, "X"), (y, "Y"), (m, "M")):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{nm} must be finite.")
    if x.sum() < 2 or (1 - x).sum() < 2:
        raise ValueError(f"Need at least 2 units per exposure arm; got {int(x.sum())} and {int((1 - x).sum())}.")

    def point(idx):
        xs, ys, ms = x[idx], y[idx], m[idx]
        cs = None if cov is None else cov[idx]
        base = ms if cs is None else np.column_stack([ms, cs])
        g = _logit(base, xs)
        gm = g[1 : 1 + ms.shape[1]]
        w = np.ones(xs.size)
        exposed = xs == 1
        w[exposed] = np.exp(-(ms[exposed] @ gm))
        Xt = xs.reshape(-1, 1) if cs is None else np.column_stack([xs, cs])
        total = _logit(Xt, ys)[1]
        direct = _wlogit(Xt, ys, w)[1]
        return float(total), float(direct)

    total, direct = point(np.arange(n))
    indirect = total - direct

    se = ci_lo = ci_hi = pval = None
    if B and B > 0:
        rng = np.random.default_rng(seed)
        draws = np.full((int(B), 3), np.nan)
        for b in range(int(B)):
            idx = rng.integers(0, n, n)
            try:
                t, d = point(idx)
                draws[b] = (t, d, t - d)
            except Exception:
                continue
        good = draws[np.all(np.isfinite(draws), axis=1)]
        if good.shape[0] >= 2:
            keys = ("total", "direct", "indirect")
            se = dict(zip(keys, map(float, good.std(axis=0, ddof=1))))
            lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
            ci_lo = dict(zip(keys, map(float, lo)))
            ci_hi = dict(zip(keys, map(float, hi)))
            pval = {
                k: float(2 * min((good[:, i] <= 0).mean(), (good[:, i] >= 0).mean()))
                for i, k in enumerate(keys)
            }

    return RichResult(
        title="Binary-outcome mediation (inverse odds-ratio weighting)",
        payload={
            "total": total,
            "direct": direct,
            "indirect": indirect,
            "or_total": float(np.exp(total)),
            "or_direct": float(np.exp(direct)),
            "or_indirect": float(np.exp(indirect)),
            "proportion_mediated": float(indirect / total) if total != 0 else np.nan,
            "se": se,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "p_value": pval,
            "n": int(n),
            "B": int(B),
            "method": "Inverse odds-ratio weighting (Tchetgen Tchetgen 2013), log-odds scale",
        },
    )


def cheatsheet():
    return "binmed: binary-outcome causal mediation by inverse odds-ratio weighting"
