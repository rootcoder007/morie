# morie.fn -- function file (rootcoder007/morie)
"""Variance Inflation Factor."""

from collections.abc import Sequence
from typing import Union

import numpy as np

from ._richresult import RichResult


def vif(X: Union[Sequence, np.ndarray]) -> RichResult:
    """Variance Inflation Factor for each column of X.

    Regress each column on the remaining columns *with an intercept*;
    VIF_j = 1 / (1 - R_j^2). The intercept matters: R_j^2 is measured
    against the mean of column j, so omitting it leaves the residual sum
    of squares incomparable with the total and inflates every VIF.

    At least two predictors are required. A VIF states how far one column
    is explained by the others, which is undefined for a single column.

    References
    ----------
    Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression
    Diagnostics: Identifying Influential Data and Sources of
    Collinearity*. Wiley, pp. 92-94.
    """
    names = [str(c) for c in getattr(X, "columns", [])]
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError("X must be two-dimensional (n observations x p predictors)")
    n, p = Xa.shape
    if p < 2:
        raise ValueError(f"VIF needs at least 2 predictors, got {p}")
    if n <= p:
        raise ValueError(f"VIF needs more observations than predictors, got n={n}, p={p}")
    if not names:
        names = [f"X{j + 1}" for j in range(p)]

    out = np.zeros(p)
    for j in range(p):
        y = Xa[:, j]
        Xo = np.column_stack([np.ones(n), np.delete(Xa, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum(resid**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[j] = 1.0 / max(1.0 - r2, 1e-12)

    jmax = int(np.argmax(out))
    return RichResult(
        title="Variance Inflation Factors",
        payload={
            "value": {nm: float(v) for nm, v in zip(names, out)},
            "vif": out,
            "extra": {
                "max_vif": float(out[jmax]),
                "max_term": names[jmax],
                "n": n,
                "p": p,
                "method": "OLS with intercept, VIF = 1 / (1 - R^2)",
            },
        },
    )


# Back-compat alias -- older imports reference `variance_inflation`.
variance_inflation = vif
