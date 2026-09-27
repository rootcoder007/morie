# morie.fn -- function file (rootcoder007/morie)
"""Least angle regression -- Efron et al. (2004), ESL Sec 3.4.4."""

from __future__ import annotations

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["esl_least_angle_reg"]


def esl_least_angle_reg(X, y, max_steps=None, standardize=True, method="lar"):
    r"""Compute the LAR coefficient path.

    LAR starts from :math:`\hat\beta = 0` and repeatedly moves the current
    coefficients toward the least-squares fit *on the active set only*,
    stopping the moment some inactive predictor's correlation with the
    residual ties the active ones -- at which point that predictor joins.
    The defining invariant is that every active predictor keeps exactly the
    same absolute correlation with the residual throughout:

    .. math::
        \left| x_j^\top (y - X\hat\beta) \right| = c
        \quad \text{for all } j \in \mathcal{A}.

    That equiangularity is asserted in the doctest, because it is the one
    property that distinguishes LAR from ordinary forward stepwise (which
    takes the full least-squares step and destroys it).

    This is LAR proper, without the lasso modification, so coefficients never
    leave the active set once they enter.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``. Standardised internally by default.
    y : array-like
        Response ``(n,)``, centred internally.
    max_steps : int, optional
        Number of LAR steps. Defaults to ``min(p, n - 1)``.
    standardize : bool
        Scale each column to unit norm before fitting. Correlations are not
        comparable across differently-scaled predictors, so turning this off
        makes the entry order depend on the units.

    method : {"lar", "lasso"}
        ``"lasso"`` applies the lasso modification (ESL Alg. 3.2a): when an
        active coefficient would cross zero the step stops there and the
        variable leaves the active set, giving the whole lasso path (piecewise
        linear in lambda, ESL Ex. 3.27) as ``lars(type = "lasso")``.

    Returns
    -------
    RichResult
        ``coef_path`` ``(steps+1, p)`` in original units, ``active`` (entry
        order), ``correlations`` at each step, ``intercept``, ``r_squared``.

    References
    ----------
    Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least
        angle regression. *Annals of Statistics*, 32(2), 407-499.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Predictors enter in order of marginal correlation, strongest first.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 5))
    >>> y = 5 * X[:, 2] - 2 * X[:, 0] + rng.normal(0, 0.1, 120)
    >>> r = esl_least_angle_reg(X, y)
    >>> [int(v) for v in r["active"][:2]]
    [2, 0]

    Equiangularity -- the active predictors keep equal absolute correlation
    with the residual, which forward stepwise would not.

    >>> c = r["correlations"][2]          # after two variables have entered
    >>> act = r["active"][:2]
    >>> bool(np.ptp(np.abs(c[act])) < 1e-8)
    True

    The path starts at zero and ends at the least-squares fit on the
    variables that entered.

    >>> bool(np.all(r["coef_path"][0] == 0))
    True
    >>> bool(r["r_squared"] > 0.99)
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if n != y.size:
        raise ValueError(f"X has {n} rows but y has {y.size}")
    if method not in ("lar", "lasso"):
        raise ValueError("method must be 'lar' or 'lasso'")
    cap = min(p, n - 1) if method == "lar" else 8 * min(p, n - 1)
    max_steps = cap if max_steps is None else int(max_steps)
    if not 1 <= max_steps <= cap:
        raise ValueError(f"max_steps must be between 1 and {cap}")

    xbar = X.mean(axis=0)
    Xc = X - xbar
    scale = np.sqrt((Xc**2).sum(axis=0)) if standardize else np.ones(p)
    scale = np.where(scale > 0, scale, 1.0)
    Xs = Xc / scale
    ybar = float(y.mean())
    yc = y - ybar

    beta = np.zeros(p)
    mu = np.zeros(n)
    active: list[int] = []
    path = [beta.copy()]
    cors = [Xs.T @ yc]
    dropped = []
    just_dropped = -1

    for _ in range(max_steps):
        c = Xs.T @ (yc - mu)
        C = np.max(np.abs(c))
        if C < 1e-12:
            break
        for j in np.flatnonzero(np.abs(np.abs(c) - C) < 1e-10):
            if j not in active and int(j) != just_dropped:
                active.append(int(j))
        just_dropped = -1
        A = np.array(active)
        s = np.sign(c[A])
        XA = Xs[:, A] * s
        G = XA.T @ XA
        try:
            Ginv1 = np.linalg.solve(G, np.ones(len(A)))
        except np.linalg.LinAlgError:
            Ginv1 = np.linalg.lstsq(G, np.ones(len(A)), rcond=None)[0]
        AA = 1.0 / np.sqrt(np.sum(Ginv1))
        w = AA * Ginv1
        u = XA @ w
        a = Xs.T @ u

        inactive = np.setdiff1d(np.arange(p), A)
        if inactive.size == 0:
            gamma = C / AA
        else:
            cand = np.r_[(C - c[inactive]) / (AA - a[inactive]),
                         (C + c[inactive]) / (AA + a[inactive])]
            cand = cand[(cand > 1e-12) & np.isfinite(cand)]
            gamma = float(cand.min()) if cand.size else C / AA

        drop = None
        if method == "lasso":
            # lasso modification: stop where an active coefficient reaches zero (Efron et al. 2004, eq 3.5)
            d = w * s
            with np.errstate(divide="ignore", invalid="ignore"):
                gt = np.where(d != 0, -beta[A] / d, np.inf)
            ok = gt > 1e-12
            if bool(np.any(ok)) and float(np.min(gt[ok])) < gamma:
                k = int(np.flatnonzero(ok & (gt == np.min(gt[ok])))[0])
                gamma = float(gt[k])
                drop = int(A[k])
        mu = mu + gamma * u
        beta[A] += gamma * w * s
        if drop is not None:
            beta[drop] = 0.0
            active.remove(drop)
            dropped.append(drop)
            just_dropped = drop
        path.append(beta.copy())
        cors.append(Xs.T @ (yc - mu))

    coef = np.array(path) / scale
    fitted = X @ coef[-1] + (ybar - xbar @ coef[-1])
    ss_tot = float(np.sum((y - ybar) ** 2))
    return RichResult(
        title="Least angle regression",
        summary_lines=[("n", n), ("p", p), ("steps", len(path) - 1),
                       ("active", len(active))],
        payload={
            "coef_path": coef, "coef": coef[-1],
            "intercept": float(ybar - xbar @ coef[-1]),
            "active": np.array(active, dtype=int),
            "correlations": np.array(cors),
            "fitted": fitted,
            "r_squared": float(1 - np.sum((y - fitted) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
            "n_steps": len(path) - 1,
            "dropped": np.array(dropped, dtype=int),
            "method": "esl_least_angle_reg" if method == "lar" else "esl_least_angle_reg (lasso modification)",
        },
    )


def cheatsheet():
    return "esllar: LAR path; active predictors keep EQUAL |corr| with the residual -- that is the invariant to check"
