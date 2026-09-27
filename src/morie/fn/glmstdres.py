"""Pearson and standardized Pearson residuals for binomial and Poisson GLMs.

Bilder & Loughin (2025), Analysis of Categorical Data with R, Sec 5.2.1.
"""

import math

from ._richresult import RichResult
from .glmprofci import _irls, _prep, _solve

__all__ = ["glmstdres"]


def glmstdres(y, X, family="binomial", trials=None):
    r"""e_m = (y_m - yhat_m)/sqrt(Var-hat(Y_m)); r_m = e_m / sqrt(1 - h_m).

    Var-hat(Y_m) is n_m pi_m (1 - pi_m) for a binomial and mu_m for a Poisson
    model; h_m are the diagonal elements of the hat matrix
    W^{1/2} X (X'WX)^{-1} X' W^{1/2}. |r_m| beyond 2 or 3 flags a poorly fitted
    explanatory variable pattern.

    Parameters
    ----------
    y : sequence
    X : n x p nested sequence
    family : {"binomial", "poisson"}
    trials : sequence, optional

    Returns
    -------
    RichResult
        Keys: fitted, pearson, standardized, hat, beta, deviance (2.10), pearson_chisq
        (5.5; the sum of squared Pearson residuals), df (n - p).

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Sec 5.2.1.

    Examples
    --------
    >>> r = glmstdres([2, 3, 6, 7, 8, 9, 10, 12, 15], [[1, x] for x in range(9)], family="poisson")
    >>> round(sum(r["hat"]), 10)
    2.0
    """
    y, X, trials = _prep(y, X, family, trials)
    p = len(X[0])
    beta, mu, _, info = _irls(y, X, family, trials, [0.0] * len(y))
    inv = [_solve(info, [float(a == b) for a in range(p)]) for b in range(p)]
    var = [m * (1 - m / ni) for m, ni in zip(mu, trials)] if family == "binomial" else mu[:]
    hat = [v * sum(x[a] * inv[a][b] * x[b] for a in range(p) for b in range(p)) for x, v in zip(X, var)]
    e = [(yi - m) / math.sqrt(v) for yi, m, v in zip(y, mu, var)]
    r = [ei / math.sqrt(1 - h) for ei, h in zip(e, hat)]
    if family == "binomial":
        dev = 2 * sum(
            (yi * math.log(yi / m) if yi > 0 else 0.0)
            + ((ni - yi) * math.log((ni - yi) / (ni - m)) if yi < ni else 0.0)
            for yi, m, ni in zip(y, mu, trials)
        )
    else:
        dev = 2 * sum((yi * math.log(yi / m) if yi > 0 else 0.0) - (yi - m) for yi, m in zip(y, mu))
    return RichResult(
        title="GLM residuals",
        summary_lines=[("max |r|", max(abs(v) for v in r))],
        payload={
            "fitted": mu,
            "pearson": e,
            "standardized": r,
            "hat": hat,
            "beta": beta,
            "deviance": dev,
            "pearson_chisq": sum(v * v for v in e),
            "df": len(y) - p,
        },
    )


def cheatsheet():
    return "glmstdres: Pearson and standardized Pearson residuals with hat values for binomial/Poisson GLMs. Bilder & Loughin Sec 5.2.1."
