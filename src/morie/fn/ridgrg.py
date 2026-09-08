# morie.fn -- k02 batch (rootcoder007/morie)
"""Ridge regression (Hoerl-Kennard) in the MASS::lm.ridge parameterisation.

Source consulted: Hoerl, A.E. and Kennard, R.W. (1970), Ridge regression:
biased estimation for nonorthogonal problems, *Technometrics* 12(1), 55-67,
equation (3.4): beta(lambda) = (X'X + lambda I)^-1 X' y.  The scaling matters
and is easy to get wrong, so this follows the convention of ``MASS::lm.ridge``
exactly: y is centred, each predictor is centred and divided by its
root-mean-square deviation (divisor n, not n - 1), the penalised solve is done
on that scale, the coefficients are divided back by the scale factors, and the
intercept is ``mean(y) - sum(beta * mean(X))``.  lambda = 0 therefore returns
ordinary least squares.  Verified against ``MASS::lm.ridge`` in the canonical
test.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ridge_regression"]


def ridge_regression(X, y, lam=0.0):
    """Ridge regression coefficients.

    Parameters
    ----------
    X : array-like
        Predictor matrix, n by q (no intercept column).
    y : array-like
        Response.
    lam : float, default 0.0
        Ridge penalty.

    Returns
    -------
    RichResult
        estimate (coefficients on the original scale, intercept first),
        coefficients, intercept, scales, gcv, df, lam, n, method.
    """
    m = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    n = len(yv)
    if m.shape[0] != n:
        m = m.T
    q = m.shape[1]
    xm = [float(np.mean(m[:, j])) for j in range(q)]
    xc = np.column_stack([m[:, j] - xm[j] for j in range(q)])
    xs = [float(np.sqrt(float(np.sum(xc[:, j] ** 2)) / n)) for j in range(q)]
    xsc = np.column_stack([xc[:, j] / xs[j] for j in range(q)])
    ym = float(np.mean(yv))
    yc = yv - ym
    xtx = np.dot(xsc.T, xsc)
    a = xtx + float(lam) * np.eye(q)
    bs = np.linalg.solve(a, np.dot(xsc.T, yc))
    beta = np.asarray([float(bs[j]) / xs[j] for j in range(q)], dtype=float)
    b0 = ym - float(np.sum(beta * np.asarray(xm, dtype=float)))
    hatd = np.dot(xsc, np.linalg.solve(a, xsc.T))
    df = float(np.trace(hatd))
    resid = yc - np.dot(xsc, bs)
    gcv = float(np.sum(resid * resid)) / (n * (1.0 - df / n) ** 2)
    return RichResult(
        payload={
            "estimate": [b0] + beta.tolist(),
            "coefficients": beta.tolist(),
            "intercept": b0,
            "scales": xs,
            "gcv": gcv,
            "df": df,
            "lam": float(lam),
            "n": int(n),
            "method": "Ridge regression, lm.ridge scaling (Hoerl & Kennard 1970, eq. 3.4)",
        }
    )


# CANONICAL TEST
# >>> X = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7], [9, 10], [10, 9]]
# >>> y = [1.2, 2.3, 2.9, 4.1, 5.2, 5.8, 7.3, 8.1, 8.9, 10.2]
# >>> r = ridge_regression(X, y, 0.5)
# >>> assert abs(r["intercept"] - 0.291850723533891) < 1e-10     # MASS::lm.ridge
# >>> assert abs(r["coefficients"][0] - 0.756531627873009) < 1e-10
# >>> assert abs(r["coefficients"][1] - 0.208586422393557) < 1e-10


def cheatsheet():
    return "ridgrg(X, y, lam): ridge regression (lm.ridge scaling)."


ridgeregression = ridge_regression
