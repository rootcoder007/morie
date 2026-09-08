# morie.fn -- function file (rootcoder007/morie)
"""Cramer-Rao lower bound from a Fisher information matrix.

Sources CONSULTED: Rao, C. R. (1945), "Information and the accuracy
attainable in the estimation of statistical parameters", *Bulletin of
the Calcutta Mathematical Society* 37:81-91; Cramer, H. (1946),
*Mathematical Methods of Statistics*, Princeton University Press.
Neither could be obtained in full (a 1945 society bulletin and a 1946
monograph); the statement implemented is the standard published one for
an unbiased estimator of a vector parameter,

    Cov(theta_hat)  >=  I(theta)^-1      (Loewner order)

so that the bound on the variance of the k-th component is the k-th
diagonal entry of the inverse information matrix, and the bound on any
linear combination a'theta is a' I^-1 a.

The inverse is formed by Gauss-Jordan elimination with partial pivoting,
in this module rather than through a shared core, so that the R mirror
runs the identical arithmetic.
"""

import math

from ._richresult import RichResult

__all__ = ["cramer_rao_bound"]


def _as_matrix(a):
    """Accept a scalar, a 1-d sequence (read as a diagonal) or a 2-d one."""
    if isinstance(a, (int, float)):
        return [[float(a)]]
    rows = list(a)
    if not rows:
        raise ValueError("fisher_info is empty")
    if isinstance(rows[0], (int, float)):
        k = len(rows)
        return [[float(rows[i]) if i == j else 0.0 for j in range(k)]
                for i in range(k)]
    out = [[float(v) for v in r] for r in rows]
    k = len(out)
    for r in out:
        if len(r) != k:
            raise ValueError("fisher_info must be square")
    return out


def _inv(A):
    """Inverse by Gauss-Jordan with partial pivoting."""
    k = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(k)]
         for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("Fisher information matrix is singular")
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(k):
            if r == c:
                continue
            fac = M[r][c] / pv
            if fac == 0.0:
                continue
            for t in range(c, 2 * k):
                M[r][t] -= fac * M[c][t]
    return [[M[i][k + j] / M[i][i] for j in range(k)] for i in range(k)]


def cramer_rao_bound(fisher_info, var_estimate=None):
    """Cramer-Rao lower bound implied by a Fisher information matrix.

    Parameters
    ----------
    fisher_info : float, or sequence, or sequence of sequences
        The Fisher information.  A scalar is a one-parameter
        information; a flat sequence is read as the diagonal of a
        diagonal information matrix; a nested sequence is the full
        square matrix.
    var_estimate : sequence, optional
        Actual variances of an estimator, one per parameter.  When
        given, the efficiency ``bound_k / var_estimate_k`` is returned
        alongside; it cannot exceed 1 for an unbiased estimator.

    Returns
    -------
    RichResult
        Keys ``bound`` (the inverse information matrix), ``variance``
        (its diagonal), ``se``, ``information``, ``k``, ``efficiency``,
        ``attained``, ``method``.
    """
    info = _as_matrix(fisher_info)
    k = len(info)
    for i in range(k):
        for j in range(i + 1, k):
            if abs(info[i][j] - info[j][i]) > 1e-12 * (1.0 + abs(info[i][j])):
                raise ValueError("Fisher information matrix must be symmetric")
    for i in range(k):
        if info[i][i] <= 0.0:
            raise ValueError("Fisher information has a non-positive diagonal")
    bound = _inv(info)
    var = [bound[i][i] for i in range(k)]
    for v in var:
        if v <= 0.0:
            raise ValueError("inverse information has a non-positive diagonal")
    se = [math.sqrt(v) for v in var]
    eff = None
    attained = None
    if var_estimate is not None:
        ve = [float(v) for v in
              ([var_estimate] if isinstance(var_estimate, (int, float))
               else list(var_estimate))]
        if len(ve) != k:
            raise ValueError("var_estimate must have one entry per parameter")
        eff = [var[i] / ve[i] for i in range(k)]
        attained = all(e <= 1.0 + 1e-9 for e in eff)
    return RichResult(
        payload={
            "bound": bound,
            "variance": var,
            "se": se,
            "information": info,
            "k": k,
            "efficiency": eff,
            "attained": attained,
            "method": "Cramer-Rao lower bound (inverse Fisher information)",
        }
    )


def cheatsheet():
    return "crmrlb: Cramér-Rao lower bound"
