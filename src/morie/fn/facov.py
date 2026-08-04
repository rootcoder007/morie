# morie.fn -- slice s04 (rootcoder007/morie)
"""Factor analytic covariance structure for multi-environment trials.

NOT IN THE BOOK.  Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer, was searched in full -- all seventeen page-range
volumes and the index, [Pages 683-691].  The phrase "factor analytic"
occurs exactly once, in the front matter, volume [Pages i-xxiv]: "for
the traits or environments, unstructured or factor analytic
variance-covariance matrices can be chosen".  The structure is named
there and nowhere specified; Chapter 5, volume [Pages 141-170], carries
only the unstructured and diagonal forms.

The structure is therefore taken from the primary source, Smith, A.,
Cullis, B. and Thompson, R. (2001), Analyzing variety by environment
data using multiplicative mixed models and adjustments for spatial field
trends, *Biometrics* 57(4), 1138-1147, which writes the k-factor variety
by environment variance matrix as

    Sigma_g = Lambda Lambda' + Psi,

Lambda the n_env-by-k loadings and Psi the diagonal of environment
specific variances.  The identifiability constraint the same paper
imposes -- the upper triangle of Lambda above the diagonal is zero --
fixes the free parameter count at n_env*k - k(k-1)/2 + n_env.

DETERMINISM.  When no loadings are supplied the canonical Lambda is laid
out on van der Corput points, not drawn, so both arms build the same
matrix.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["factor_analytic_covariance"]


def factor_analytic_covariance(n_env, n_factors, loadings=None, psi=None):
    """Sigma = Lambda Lambda' + Psi, the k-factor analytic structure.

    Parameters
    ----------
    n_env : int
        Number of environments (or traits), the order of Sigma.
    n_factors : int
        k, the number of factors; 0 <= k <= n_env.
    loadings : array-like, optional
        n_env-by-k loading matrix.  When absent a deterministic
        low-discrepancy Lambda is used, in the lower-triangular
        parameterisation Smith et al. impose.
    psi : array-like, optional
        The n_env specific variances; unit variances when absent.

    Returns
    -------
    estimate  : Sigma[0][0]
    Sigma     : the covariance matrix
    Lambda    : the loadings actually used
    Psi       : the specific variances actually used
    n_params  : n_env*k - k(k-1)/2 + n_env
    """
    m = int(n_env)
    kk = int(n_factors)
    if m < 1:
        raise ValueError("factor_analytic_covariance: n_env must be at least 1")
    if kk < 0 or kk > m:
        raise ValueError("factor_analytic_covariance: n_factors must lie between 0 and n_env")
    if loadings is None:
        L = [[0.0] * kk for _ in range(m)]
        for i in range(m):
            for j in range(kk):
                if j <= i:                       # Smith et al. identifiability
                    L[i][j] = core.vdc(i * kk + j, 2 + j) + 0.5
    else:
        L = core.mat(loadings)
        if len(L) != m or (kk > 0 and len(L[0]) != kk):
            raise ValueError("factor_analytic_covariance: loadings must be n_env by n_factors")
    if psi is None:
        P = [1.0] * m
    else:
        P = core.vec(psi)
        if len(P) != m:
            raise ValueError("factor_analytic_covariance: psi must have n_env entries")
        for v in P:
            if v < 0.0:
                raise ValueError("factor_analytic_covariance: specific variances must be non-negative")
    S = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            s = 0.0
            for j in range(kk):
                s += L[a][j] * L[b][j]
            S[a][b] = s + (P[a] if a == b else 0.0)
    return RichResult(
        title="Factor analytic covariance",
        summary_lines=[("environments", m), ("factors", kk)],
        payload={
            "estimate": S[0][0],
            "Sigma": S,
            "Lambda": L,
            "Psi": P,
            "n_params": m * kk - kk * (kk - 1) // 2 + m,
            "n": m,
            "method": "Sigma = Lambda Lambda' + Psi, Smith, Cullis and Thompson (2001); not in the book",
        },
    )


def cheatsheet():
    return "facov: Factor analytic covariance structure for multi-environment trials"
