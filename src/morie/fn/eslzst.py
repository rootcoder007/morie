# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Z-scores for coefficient significance (ESL Ch 3.2)."""

import numpy as np

from ._richresult import RichResult
from .eslsbt import esl_se_beta

__all__ = ["esl_z_score"]


def esl_z_score(X, y, beta):
    """
    z_j = beta_j / se(beta_j).

    ESL Eq. 3.12 calls these Z-scores and notes they follow t_{n-p-1}
    under the null with sigma estimated -- normal only asymptotically.
    Both tail probabilities ship: the normal one and the t one with
    the correct degrees of freedom, because for small n they differ
    materially and using the normal there overstates significance.
    A zero standard error (perfect fit) gives an infinite z, reported
    rather than masked.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    beta : array-like, shape (p,)
        Fitted coefficients.

    Returns
    -------
    result : dict
        Keys: estimate (z of the first coefficient), z, se,
        p_normal, p_t, df, n, p, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 3.2 (Eq. 3.12).

    Examples
    --------
    >>> X = [[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]]
    >>> y = [3.2, -0.8, 2.8, -1.2]
    >>> import numpy as np
    >>> b = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
    >>> out = esl_z_score(X, y, b)
    >>> len(out["z"])
    2
    >>> out["df"]
    2
    >>> out["p_t"][1] < 0.05        # slope clearly nonzero
    True
    >>> exact = esl_z_score(X, [3.0, -1.0, 3.0, -1.0], [1.0, 2.0])
    >>> exact["z"][1]
    inf
    """
    fit = esl_se_beta(X, y, beta)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    se = np.asarray(fit["se"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, beta / se, np.where(beta == 0, 0.0, np.inf * np.sign(beta)))
    df = fit["df_residual"]

    def _norm_sf(t):
        return 0.5 * (1.0 - np.math.erf(abs(t) / np.sqrt(2.0))) * 2.0 if np.isfinite(t) else 0.0

    def _t_sf(t, k):
        """Two-sided t tail via the regularised incomplete beta (stdlib-only)."""
        import math
        if not np.isfinite(t):
            return 0.0
        x = k / (k + t * t)
        # continued fraction for I_x(k/2, 1/2)
        a, b = k / 2.0, 0.5
        lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        if x <= 0:
            return 0.0
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            c = 1.0 + num / c
            if abs(c) < 1e-30:
                c = 1e-30
            f *= c * d
            if abs(1.0 - c * d) < 1e-14:
                break
        return front * (f - 1.0)

    import math
    p_norm = [float(math.erfc(abs(v) / math.sqrt(2.0))) if np.isfinite(v) else 0.0 for v in z]
    p_t = [float(_t_sf(v, df)) for v in z]
    return RichResult(payload={
        "estimate": float(z[0]), "z": [float(v) for v in z],
        "se": [float(v) for v in se], "p_normal": p_norm, "p_t": p_t,
        "df": int(df), "n": fit["n"], "p": fit["p"],
        "method": "z_j = beta_j/se(beta_j); t_{n-p} and normal tails both reported"})


def cheatsheet():
    return "eslzst: z = beta/se; t tail with df = n-p is the honest one for small n"
