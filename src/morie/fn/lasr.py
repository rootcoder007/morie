# morie.fn -- function file (rootcoder007/morie)
"""Lasso regression with R-style verbose result."""

from collections.abc import Sequence
from typing import Union

from . import _array_core as np
from ._ml_core import Lasso


def lasr(
    X: Union[Sequence, np.ndarray],
    y: Union[Sequence, np.ndarray],
    alpha: float = 1.0,
    fit_intercept: bool = True,
    max_iter: int = 10000,
):
    r"""Lasso regression by cyclic coordinate descent.

    Minimises ``(1/(2n)) ||y - b0 - X b||^2 + alpha ||b||_1`` (Tibshirani
    1996, scikit-learn scaling). With an intercept the columns and ``y``
    are centred; each coordinate update is the soft-threshold ``b_j =
    S(x_j'r + b_j ||x_j||^2, n alpha) / ||x_j||^2`` (Friedman, Hastie and
    Tibshirani 2010), sweeps stopping when the largest coefficient change
    is below ``1e-8``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors (not standardised here).
    y : array-like, shape (n,)
        Response.
    alpha : float
        L1 penalty.
    fit_intercept : bool
        Centre and fit an unpenalised intercept.
    max_iter : int
        Maximum number of sweeps.

    Returns
    -------
    RichResult
        ``coef``, ``intercept``, ``r2`` (training), ``alpha``, ``nonzero``.

    References
    ----------
    Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the
    Royal Statistical Society B*, 58(1), 267-288.

    Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized
    linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

    Examples
    --------
    >>> X = [[1.0, 0.5], [2.0, -1.0], [3.0, 0.2], [4.0, 1.5], [5.0, -0.3], [6.0, 0.8]]
    >>> r = lasr(X, [1.1, 2.3, 2.8, 4.4, 4.9, 6.2], alpha=0.2)
    >>> [round(c, 6) for c in r["coef"]], round(r["intercept"], 6)
    ([0.928571, 0.0], 0.366667)
    """
    from ._richresult import RichResult

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    nonzero = int(np.sum(np.abs(coefs) > 1e-10))
    coef_rows = [[f"x{i + 1}", f"{c:.6g}", "selected" if abs(c) > 1e-10 else "zeroed"] for i, c in enumerate(coefs)]
    return RichResult(
        title="Lasso regression",
        summary_lines=[
            ("Alpha (L1)", alpha),
            ("R^2 (training)", r2),
            ("Intercept", intercept),
            ("Predictors selected", f"{nonzero} of {len(coefs)}"),
            ("n observations", len(y)),
        ],
        tables=[
            {
                "title": "Coefficients:",
                "headers": ["Predictor", "Coefficient", "Status"],
                "rows": coef_rows,
            }
        ],
        warnings=[]
        if nonzero > 0
        else [
            "all coefficients zeroed - alpha is too large for this data; "
            "try a smaller value or use cross-validated LassoCV."
        ],
        payload={"coef": coefs.tolist(), "intercept": intercept, "r2": r2, "alpha": alpha, "nonzero": nonzero},
    )


def cheatsheet() -> str:
    return "lasr: lasr(X, y, alpha, fit_intercept, max_iter) -> Lasso regression (L1-regularized OLS) - performs variable selection."
