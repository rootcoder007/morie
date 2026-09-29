# morie.fn -- function file (rootcoder007/morie)
"""Ridge regression with R-style verbose result."""

from ._qpcore import inverse, ssum
from ._richresult import RichResult


def rdgr(X, y, alpha: float = 1.0, fit_intercept: bool = True):
    r"""Ridge regression: minimise ``||y - b0 - X b||^2 + alpha ||b||^2`` (intercept unpenalised).

    With centred ``X`` and ``y`` (when ``fit_intercept``) ``b = (Xc'Xc +
    alpha I)^{-1} Xc'yc`` and ``b0 = ybar - xbar'b`` (Hoerl and Kennard
    1970; the convention of ``sklearn.linear_model.Ridge``); ``r2`` is the
    training ``R^2``.

    References
    ----------
    Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased
    estimation for nonorthogonal problems. *Technometrics* 12, 55-67.

    Examples
    --------
    >>> r = rdgr([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.1, 2.9, 4.2], alpha=1.0)
    >>> round(r["coef"][0], 12), round(r["intercept"], 12)
    (0.866666666667, 0.383333333333)
    """
    Xm = [
        [float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in (X.tolist() if hasattr(X, "tolist") else X)
    ]
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    n, p = len(Xm), len(Xm[0])
    xm = [ssum(r[j] for r in Xm) / n for j in range(p)] if fit_intercept else [0.0] * p
    ym = ssum(yv) / n if fit_intercept else 0.0
    Xc = [[r[j] - xm[j] for j in range(p)] for r in Xm]
    yc = [v - ym for v in yv]
    A = inverse([[ssum(r[a] * r[b] for r in Xc) + (alpha if a == b else 0.0) for b in range(p)] for a in range(p)])
    g = [ssum(Xc[i][a] * yc[i] for i in range(n)) for a in range(p)]
    b = [ssum(A[a][c] * g[c] for c in range(p)) for a in range(p)]
    b0 = ym - ssum(xm[j] * b[j] for j in range(p))
    fit = [b0 + ssum(r[j] * b[j] for j in range(p)) for r in Xm]
    ybar = ssum(yv) / n
    tss = ssum((v - ybar) ** 2 for v in yv)
    r2 = 1.0 - ssum((u - v) ** 2 for u, v in zip(yv, fit)) / tss if tss > 0 else 0.0
    return RichResult(
        title="Ridge regression",
        summary_lines=[
            ("Alpha (L2)", alpha),
            ("R^2 (training)", r2),
            ("Intercept", b0),
            ("n predictors", p),
            ("n observations", n),
        ],
        tables=[
            {
                "title": "Coefficients:",
                "headers": ["Predictor", "Coefficient"],
                "rows": [[f"x{i + 1}", f"{c:.6g}"] for i, c in enumerate(b)],
            }
        ],
        warnings=[] if alpha > 0 else ["alpha=0; behaves like plain OLS."],
        payload={"coef": b, "intercept": b0, "r2": r2, "alpha": alpha, "fitted": fit},
    )


def cheatsheet() -> str:
    return "rdgr: rdgr(X, y, alpha, fit_intercept) -> ridge (Xc'Xc + alpha I)^-1 Xc'yc, unpenalised intercept."
