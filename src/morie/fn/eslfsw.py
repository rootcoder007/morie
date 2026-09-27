"""Incremental forward stagewise regression (ESL Alg. 3.4 and 16.1)."""

import math

from ._richresult import RichResult

__all__ = ["esl_forward_stagewise"]


def esl_forward_stagewise(X, y, eps=0.01, max_steps=5000, standardize=True, tol=1e-12):
    r"""Forward stagewise regression :math:`FS_\varepsilon`.

    ESL Alg. 3.4: with the response centred and the predictors centred (and
    scaled to unit norm), start from :math:`\beta = 0`, :math:`r = y`; at each
    step find the predictor :math:`x_j` most correlated with r, set
    :math:`\delta_j = \varepsilon\,\mathrm{sign}\langle x_j, r\rangle`,
    :math:`\beta_j \leftarrow \beta_j + \delta_j`, :math:`r \leftarrow r - \delta_jx_j`.
    With the columns of X taken as basis functions :math:`T_k(x)` (e.g. trees)
    this is ESL Alg. 16.1 with squared-error loss. As :math:`\varepsilon\to0` the
    coefficient path tends to the infinitesimal stagewise path of
    ``lars(type = "forward.stagewise")``. Stops when the largest correlation
    is below ``tol`` or after ``max_steps``.

    Parameters
    ----------
    X : n x p nested sequence
    y : sequence of n floats
    eps : float
        Step size, > 0.
    max_steps : int
    standardize : bool
        Scale centred columns to unit norm (coefficients are returned on the
        original scale).
    tol : float

    Returns
    -------
    RichResult
        ``coef`` (original scale), ``intercept``, ``path`` (standardised
        coefficients after each step), ``chosen`` (variable per step),
        ``l1_norm``, ``steps``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), secs. 3.8.1 and 16.2.
    """
    rows = [[float(v) for v in r] for r in X]
    yy = [float(v) for v in y]
    n, p = len(rows), len(rows[0])
    if len(yy) != n or eps <= 0:
        raise ValueError("need X and y with matching rows and eps > 0")
    xbar = [sum(r[j] for r in rows) / n for j in range(p)]
    cols = [[r[j] - xbar[j] for r in rows] for j in range(p)]
    sc = [math.sqrt(sum(v * v for v in c)) if standardize else 1.0 for c in cols]
    sc = [s if s > 0 else 1.0 for s in sc]
    cols = [[v / s for v in c] for c, s in zip(cols, sc)]
    ybar = sum(yy) / n
    r = [v - ybar for v in yy]
    beta = [0.0] * p
    path, chosen = [beta[:]], []
    for _ in range(int(max_steps)):
        cor = [sum(a * b for a, b in zip(c, r)) for c in cols]
        j = max(range(p), key=lambda k: (abs(cor[k]), -k))
        if abs(cor[j]) < tol:
            break
        d = eps if cor[j] > 0 else -eps
        beta[j] += d
        r = [ri - d * cj for ri, cj in zip(r, cols[j])]
        path.append(beta[:])
        chosen.append(j)
    coef = [b / s for b, s in zip(beta, sc)]
    return RichResult(
        title="Forward stagewise regression",
        summary_lines=[("steps", len(chosen))],
        payload={
            "coef": coef,
            "intercept": ybar - sum(c * m for c, m in zip(coef, xbar)),
            "path": path,
            "chosen": chosen,
            "l1_norm": sum(abs(b) for b in beta),
            "steps": len(chosen),
        },
    )


def cheatsheet():
    return "eslfsw: move the most-correlated coefficient by eps sign(<x_j, r>) (ESL Alg. 3.4, 16.1)"
