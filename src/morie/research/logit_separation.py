# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P5: separation in a logistic fit (``research/lean/P5Separation.lean``; Albert & Anderson 1984).

* ``Research.P5.ll1_strictMono`` / ``ll1_neg``
* ``Research.P5.loglik_lt_shift`` / ``loglik_lt_shift_quasi`` / ``no_mle`` / ``no_mle_quasi``
* ``Research.P5.loglik_neg`` / ``loglik_tendsto_zero``

R parity: ``rmorie`` ``R/logit_separation.R`` (``morie_logit_separation``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn._sci_core import linprog

__all__ = ["logit_separation"]


def logit_separation(y, x, intercept=True, method="auto") -> dict:
    """Detect separation in a logistic regression and show why the fit cannot converge.

    The exact check is the linear programme: maximise ``sum_i s_i x_i . d``
    subject to ``0 <= s_i x_i . d <= 1``; a positive optimum is a separating
    direction (complete when every margin is positive, quasi-complete when
    some are zero). ``method="glm"`` inspects an iteratively reweighted fit
    for fitted probabilities at 0 or 1 instead, a heuristic.

    Examples
    --------
    >>> x = [[1, 0.2], [1, -1], [1, 0.5], [0, 0.1], [0, -0.4], [0, 1.2], [0, 0.3]]
    >>> r = logit_separation([1, 1, 1, 0, 0, 0, 0], x)
    >>> (r["separation"], all(v < 0 for v in r["loglik_along"].values()), r["loglik_along"]["t=100"] > -1e-6)
    ('complete', True, True)
    """
    if method not in ("auto", "lp", "glm"):
        raise ValueError("method must be 'auto', 'lp' or 'glm'")
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    y = np.asarray(y, dtype=float)
    if y.shape[0] != n:
        raise ValueError("y must have one entry per row of x")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be 0/1")
    if intercept:
        X = np.column_stack([np.ones(n), X])
    p = X.shape[1]
    s = 2 * y - 1
    sx = X * s.reshape(-1, 1)

    def loglik(b):
        eta = sx @ np.asarray(b, dtype=float)
        return float(sum(-math.log1p(math.exp(-float(v))) for v in eta))

    if method == "auto":
        method = "lp"
    direction = None
    if method == "lp":
        # free d = dp - dm, both >= 0 (the solver's default bounds); maximise = minimise the negative
        cs = sx.sum(axis=0)
        c = np.concatenate([-cs, cs])
        A_ub = np.vstack([np.hstack([-sx, sx]), np.hstack([sx, -sx])])
        b_ub = np.concatenate([np.zeros(n), np.ones(n)])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub)
        if getattr(res, "success", False) and res.fun is not None and -float(res.fun) > 1e-8:
            sol = np.asarray(res.x, dtype=float)
            direction = sol[:p] - sol[p:]
    else:
        from morie.fn._glm_core import glm

        fit = glm(y, X, family="binomial", add_intercept=False, max_iter=200)
        fitted = np.asarray(fit["fitted"], dtype=float)
        if np.any((fitted < 1e-8) | (fitted > 1 - 1e-8)):
            direction = np.asarray(fit["coef"], dtype=float)
    margins = None if direction is None else sx @ direction
    sep = "none"
    n_zero = None
    if margins is not None:
        tol = 1e-10 * max(1.0, float(np.max(np.abs(margins))))
        if np.all(margins > tol):
            sep = "complete"
        elif np.all(margins > -tol) and np.any(margins > tol):
            sep = "quasi-complete"
        n_zero = int(np.sum(np.abs(margins) <= tol))
    if sep == "none":
        direction = None
        margins = None
        n_zero = None
    along = None
    if direction is not None:
        along = {f"t={t}": loglik(t * direction) for t in (0, 1, 10, 100)}
    return {
        "separation": sep,
        "direction": None if direction is None else [float(v) for v in direction],
        "margins": None if margins is None else [float(v) for v in margins],
        "n_zero_margin": n_zero,
        "loglik_along": along,
        "method": method,
        "theorems": [
            "Research.P5.loglik_lt_shift",
            "Research.P5.loglik_lt_shift_quasi",
            "Research.P5.no_mle",
            "Research.P5.no_mle_quasi",
            "Research.P5.loglik_neg",
            "Research.P5.loglik_tendsto_zero",
        ],
    }
