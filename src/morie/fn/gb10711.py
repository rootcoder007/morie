# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic covariance of the k-sample control vector -- Theorem 10.7.1."""

import math

from ._richresult import RichResult

__all__ = ['kctrlasymp', 'gibbons_ctrl_normal_asymp']


def kctrlasymp(lam, dens, pval):
    """Covariance matrix Sigma of the limiting normal for W_N.

    Theorem 10.7.1 (book p. 375).  With lambda_i the limiting sample
    fractions, f_i(theta_1) the densities at the control quantile and
    p = F_1(theta_1), the (k-1)-vector
    N^{1/2}[W_{i+1}/n_{i+1} - F_{i+1}(theta_1)] is asymptotically
    normal with mean 0 and

    .. math:: \\sigma_{ij} = \\frac{Q_i Q_j p(1-p)}{\\lambda_1}
        + \\frac{\\delta_{ij} F_{i+1}(\\theta_1)
                  [1 - F_{i+1}(\\theta_1)]}{\\lambda_{i+1}},

    where Q_i = f_{i+1}(theta_1) / f_1(theta_1).

    Parameters
    ----------
    lam : sequence of float
        lambda_1, ..., lambda_k, each in (0, 1).
    dens : sequence of float
        f_1(theta_1), ..., f_k(theta_1), all strictly positive.
    pval : sequence of float
        p = F_1(theta_1) first, then F_2(theta_1), ..., F_k(theta_1).

    Returns
    -------
    RichResult
        keys ``sigma`` (the (k-1) x (k-1) matrix as nested lists),
        ``q`` (the Q_i), ``p``, ``k``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Theorem 10.7.1, p. 375.
    """
    lam = [float(v) for v in lam]
    dens = [float(v) for v in dens]
    fv = [float(v) for v in pval]
    k = len(lam)
    if k < 2:
        raise ValueError("need at least 2 populations.")
    if len(dens) != k or len(fv) != k:
        raise ValueError("lam, dens and pval must have equal length k.")
    if any(not 0.0 < v < 1.0 for v in lam):
        raise ValueError("every lambda must lie strictly inside (0, 1).")
    if any(v <= 0.0 for v in dens):
        raise ValueError("densities must be strictly positive.")
    p = fv[0]
    qs = [dens[i] / dens[0] for i in range(1, k)]
    sig = []
    for i in range(k - 1):
        row = []
        for j in range(k - 1):
            v = qs[i] * qs[j] * p * (1.0 - p) / lam[0]
            if i == j:
                v += fv[i + 1] * (1.0 - fv[i + 1]) / lam[i + 1]
            row.append(float(v))
        sig.append(row)
    return RichResult(
        payload={
            "sigma": sig,
            "q": qs,
            "p": float(p),
            "k": int(k),
            "method": "control-vector covariance, Theorem 10.7.1",
        }
    )


gibbons_ctrl_normal_asymp = kctrlasymp
