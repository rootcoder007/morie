# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""SEM residual covariance matrix and its summary fit indices."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["sem_residual"]


def sem_residual(sample_cov, fitted_cov):
    r"""Residual matrix of a fitted structural equation model and the fit
    indices built from it.

    The raw residual matrix is the model misfit in the covariance metric,

    .. math::  E = S - \hat\Sigma(\hat\theta),

    with S the sample covariance and :math:`\hat\Sigma` the model-implied
    covariance.  Summaries over the lower triangle including the diagonal,
    :math:`p(p+1)/2` distinct elements:

    .. math::

        \mathrm{RMR} = \sqrt{\frac{2}{p(p+1)}
                       \sum_{i \le j} (s_{ij} - \hat\sigma_{ij})^2},
        \qquad
        \mathrm{SRMR} = \sqrt{\frac{2}{p(p+1)}
                       \sum_{i \le j}
                       \left(\frac{s_{ij} - \hat\sigma_{ij}}
                                  {\sqrt{s_{ii} s_{jj}}}\right)^2}.

    SRMR divides each residual by the product of the observed standard
    deviations, so it is scale free; Hu & Bentler's conventional cutoff is
    SRMR <= 0.08.  ``srmr_acceptable`` applies that cutoff.  Bentler & Yuan
    show that in small samples the residual-based statistics behave far
    better than the likelihood-ratio chi-square, which is why the residual
    matrix rather than the chi-square is the object returned here.

    Both matrices must be square, of the same order, and symmetric to
    within 1e-8; a non-symmetric input is an error, not something to
    silently symmetrise.

    Parameters
    ----------
    sample_cov : array-like
        Observed covariance matrix S, p by p.
    fitted_cov : array-like
        Model-implied covariance matrix Sigma-hat, p by p.

    Returns
    -------
    RichResult
        ``estimate`` is the SRMR.  ``residual`` is the full matrix E,
        ``max_abs_residual`` its largest element in absolute value.

    References
    ----------
    Bentler, P. M. & Yuan, K.-H. (1999). Structural equation modeling with
    small samples: test statistics. Multivariate Behavioral Research 34(2),
    181-197. doi:10.1207/s15327906mb340203
    Hu, L. & Bentler, P. M. (1999). Cutoff criteria for fit indexes in
    covariance structure analysis. Structural Equation Modeling 6(1), 1-55.
    doi:10.1080/10705519909540118
    """
    S = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(sample_cov, dtype=float)).tolist()]
    G = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(fitted_cov, dtype=float)).tolist()]
    p = len(S)
    if p == 0 or any(len(r) != p for r in S):
        raise ValueError("sem_residual: sample_cov must be square")
    if len(G) != p or any(len(r) != p for r in G):
        raise ValueError("sem_residual: fitted_cov must have the same order as sample_cov")
    for i in range(p):
        for j in range(p):
            if abs(S[i][j] - S[j][i]) > 1e-8:
                raise ValueError("sem_residual: sample_cov is not symmetric")
            if abs(G[i][j] - G[j][i]) > 1e-8:
                raise ValueError("sem_residual: fitted_cov is not symmetric")
    for i in range(p):
        if S[i][i] <= 0.0:
            raise ValueError("sem_residual: sample_cov has a non-positive variance")

    E = [[S[i][j] - G[i][j] for j in range(p)] for i in range(p)]

    m = p * (p + 1) // 2
    ss_raw = 0.0
    ss_std = 0.0
    max_abs = 0.0
    max_abs_std = 0.0
    for i in range(p):
        for j in range(i + 1):
            e = E[i][j]
            ss_raw += e * e
            z = e / math.sqrt(S[i][i] * S[j][j])
            ss_std += z * z
            if abs(e) > max_abs:
                max_abs = abs(e)
            if abs(z) > max_abs_std:
                max_abs_std = abs(z)
    rmr = math.sqrt(ss_raw / m)
    srmr = math.sqrt(ss_std / m)

    tr = sum(E[i][i] for i in range(p))
    return RichResult(
        payload={
            "estimate": srmr,
            "residual": E,
            "srmr": srmr,
            "rmr": rmr,
            "max_abs_residual": max_abs,
            "max_abs_standardised": max_abs_std,
            "sum_sq_residual": ss_raw,
            "mean_residual": tr / p,
            "trace_residual": tr,
            "n_elements": float(m),
            "p": float(p),
            "srmr_acceptable": 1.0 if srmr <= 0.08 else 0.0,
            "method": "SEM residual matrix S - Sigma-hat with RMR/SRMR (Bentler & Yuan 1999; Hu & Bentler 1999)",
        }
    )


def cheatsheet():
    return "semsro: SEM residual matrix S - Sigma-hat, with RMR and SRMR"


# compact alias per ledger/NAMING.md
semresidual = sem_residual
