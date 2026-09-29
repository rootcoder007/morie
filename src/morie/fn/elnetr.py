# morie.fn -- function file (rootcoder007/morie)
"""Elastic net regression with R-style verbose result."""

from collections.abc import Sequence
from typing import Union

from . import _array_core as np
from ._ml_core import ElasticNet


def elnetr(
    X: Union[Sequence, np.ndarray],
    y: Union[Sequence, np.ndarray],
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 10000,
):
    r"""Elastic net regression by cyclic coordinate descent.

    Minimises ``(1/(2n)) ||y - b0 - X b||^2 + alpha * l1_ratio * ||b||_1 +
    alpha * (1 - l1_ratio) / 2 * ||b||^2`` (Zou and Hastie 2005, in the
    parametrisation of scikit-learn). With an intercept the columns and
    ``y`` are centred; each coordinate update is the soft-threshold
    ``b_j = S(x_j'r + b_j ||x_j||^2, n alpha l1_ratio) / (||x_j||^2 + n
    alpha (1 - l1_ratio))`` (Friedman, Hastie and Tibshirani 2010), sweeps
    stopping when the largest coefficient change is below ``1e-8``.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Predictors (not standardised here).
    y : array-like, shape (n,)
        Response.
    alpha : float
        Overall penalty.
    l1_ratio : float
        L1 share of the penalty (1 = lasso, 0 = ridge).
    fit_intercept : bool
        Centre and fit an unpenalised intercept.
    max_iter : int
        Maximum number of sweeps.

    Returns
    -------
    RichResult
        ``coef``, ``intercept``, ``r2`` (training), ``alpha``,
        ``l1_ratio``, ``nonzero``.

    References
    ----------
    Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net.
    *Journal of the Royal Statistical Society B*, 67(2), 301-320.

    Friedman, J., Hastie, T. and Tibshirani, R. (2010). Regularization paths for generalized
    linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.

    Examples
    --------
    >>> X = [[1.0, 0.5], [2.0, -1.0], [3.0, 0.2], [4.0, 1.5], [5.0, -0.3], [6.0, 0.8]]
    >>> r = elnetr(X, [1.1, 2.3, 2.8, 4.4, 4.9, 6.2], alpha=0.1, l1_ratio=0.5)
    >>> [round(c, 6) for c in r["coef"]], round(r["intercept"], 6)
    ([0.960139, 0.024295], 0.249296)
    """
    from ._richresult import RichResult

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    nonzero = int(np.sum(np.abs(coefs) > 1e-10))
    coef_rows = [[f"x{i + 1}", f"{c:.6g}", "selected" if abs(c) > 1e-10 else "zeroed"] for i, c in enumerate(coefs)]
    return RichResult(
        title="Elastic net regression",
        summary_lines=[
            ("Alpha", alpha),
            ("L1 ratio (rho)", l1_ratio),
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
        interpretation=(
            f"l1_ratio={l1_ratio}: "
            + (
                "pure Ridge (no selection)"
                if l1_ratio == 0
                else "pure Lasso (full selection)"
                if l1_ratio == 1
                else f"mix - {l1_ratio * 100:.0f}% L1 / {(1 - l1_ratio) * 100:.0f}% L2"
            )
        ),
        payload={
            "coef": coefs.tolist(),
            "intercept": intercept,
            "r2": r2,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "nonzero": nonzero,
        },
    )


def cheatsheet() -> str:
    return "elnetr: elnetr(X, y, alpha, l1_ratio, fit_intercept, max_iter) -> Elastic net - convex combination of L1 and L2 penalties."
