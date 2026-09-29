# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit marginal effects."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult


def _rows(A):
    rows = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]


def _impacts(coef, rho, X, W, link, method):
    """Average direct / indirect / total impacts of a SAR binary-choice model (non-constant covariates)."""
    b = [float(v) for v in coef]
    Xm, Wm = _rows(X), _rows(W)
    n, p = len(Xm), len(b)
    Si = inverse([[(1.0 if i == j else 0.0) - float(rho) * Wm[i][j] for j in range(n)] for i in range(n)])
    xb = [ssum(r[k] * b[k] for k in range(p)) for r in Xm]
    eta = [ssum(Si[i][j] * xb[j] for j in range(n)) for i in range(n)]
    if method == "marginal":
        sd = [math.sqrt(ssum(Si[i][m] * Si[i][m] for m in range(n))) for i in range(n)]
    elif method == "lesage_pace":
        sd = [1.0] * n
    else:
        raise ValueError("method must be 'lesage_pace' or 'marginal'")
    z = [e / s for e, s in zip(eta, sd)]
    if link == "probit":
        dens = [math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi) for v in z]
    else:
        dens = [1.0 / (2.0 + math.exp(v) + math.exp(-v)) for v in z]
    dd = [d / s for d, s in zip(dens, sd)]
    diag = ssum(dd[i] * Si[i][i] for i in range(n)) / n
    tot = ssum(dd[i] * ssum(Si[i]) for i in range(n)) / n
    cols = [k for k in range(p) if len({r[k] for r in Xm}) > 1]
    direct = [diag * b[k] for k in cols]
    total = [tot * b[k] for k in cols]
    return RichResult(
        payload={
            "direct": direct,
            "indirect": [t - d for t, d in zip(total, direct)],
            "total": total,
            "observation_direct": [[dd[i] * Si[i][i] * b[k] for k in cols] for i in range(n)],
            "variables": cols,
        }
    )


def spprmf(coef, rho, X, W, method="lesage_pace"):
    r"""Marginal effects (impacts) of the SAR probit model y* = rho W y* + X beta + e.

    With S = I - rho W and eta = S^{-1} X beta the effect of covariate
    r is the n x n matrix diag(phi(eta)) S^{-1} beta_r; its mean
    diagonal is the average direct effect, its mean row sum the average total
    effect and their difference the indirect (spillover) effect (LeSage and
    Pace 2009, sec. 10.1.6; spatialprobit::marginal.effects, computed
    exactly rather than by the trace series). method="marginal" uses the
    derivative of the exact marginal probability Phi(eta_i / s_i), s_i^2
    = ((S'S)^{-1})_ii. extra rows per observation give the direct
    effects; constant columns are skipped.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> r = spprmf([-0.2, 0.9], 0.4, X, W)
    >>> round(r["direct"][0], 10), round(r["total"][0], 10)
    (0.2980559897, 0.4507319009)
    """
    return _impacts(coef, rho, X, W, "probit", method)


spprmf_fn = spprmf


def cheatsheet() -> str:
    return "spprmf(coef, rho, X, W) -> SAR probit average direct/indirect/total impacts (LeSage and Pace 2009)."
