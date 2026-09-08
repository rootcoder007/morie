"""Rao-Scott corrected chi-square for complex surveys (Rao & Scott 1981)."""

import math

from . import _array_core as np
from ._stats_core import chi2
from ._richresult import RichResult

__all__ = ["raoscot", "rao_scott_chisq"]


def raoscot(p_hat, p0, n, V=None, deffs=None):
    """
    First-order Rao-Scott corrected goodness-of-fit chi-square.

    Rao & Scott (1981): under a complex design the Pearson statistic
    X^2 = n sum_i (p_hat_i - p0_i)^2 / p0_i is asymptotically a
    weighted sum of chi-square(1) variables whose weights lambda_i
    are the eigenvalues of D = P^{-1} V (their generalized design
    effects; P = diag(p) - p p').  Their first-order correction
    treats X^2 / lambda_bar as chi-square with k - 1 degrees of
    freedom, where lambda_bar is the mean generalized deff -- exact
    when V is supplied (mean eigenvalue of P0^{-1} V restricted to
    the k-1 dimensional simplex space, computed here as
    trace(P0^{-1} V)/(k-1)), or the paper's cell-deff estimate
    lambda_hat = sum_i (1 - p0_i) d_i / (k - 1) when only cell
    design effects d_i are known.  Their Section 2.3 model result:
    if V = c P0 (uniform clustering a la Cohen/Altham/Brier), then
    every eigenvalue is c and X^2 / c is exactly chi-square_{k-1}.

    Sources
    -------
    Rao, J. N. K. & Scott, A. J. (1981). The analysis of categorical
    data from complex sample surveys: chi-squared tests for goodness
    of fit and independence in two-way tables. *JASA*, 76(374),
    221-230, Secs. 2-3 (local copy fetched-wave3/The Analysis of
    Categorical Data from Complex Sample Surveys...pdf).

    Parameters
    ----------
    p_hat : sequence of float
        Estimated cell proportions (sum to 1).
    p0 : sequence of float
        Hypothesized proportions (positive, sum to 1).
    n : int
        Sample size.
    V : matrix, optional
        Estimated n * covariance matrix of p_hat (k x k); enables
        the trace-based mean generalized deff.
    deffs : sequence of float, optional
        Cell design effects d_i (used when V is absent).

    Returns
    -------
    RichResult
        Keys: statistic (X^2), corrected (X^2 / lambda_bar),
        lambda_bar, df, p_value (of the corrected statistic).
    """
    ph = [float(v) for v in p_hat]
    p0v = [float(v) for v in p0]
    k = len(ph)
    if len(p0v) != k or k < 2:
        raise ValueError("p_hat and p0 must be paired with k >= 2")
    if any(v <= 0 for v in p0v):
        raise ValueError("p0 must be positive")
    if abs(sum(ph) - 1.0) > 1e-6 or abs(sum(p0v) - 1.0) > 1e-6:
        raise ValueError("proportions must sum to 1")
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    x2 = n * sum((a - b) ** 2 / b for a, b in zip(ph, p0v))
    df = k - 1
    if V is not None:
        Vm = [[float(v) for v in row] for row in V]
        # lambda_bar = trace(P0^+ V) / (k - 1); on the simplex the
        # Moore-Penrose action of P0 = diag(p0) - p0 p0' reduces to
        # sum_i V_ii / p0_i - sum_ij V_ij  (since P0^+ acts as
        # diag(1/p0) minus the constant direction, and rows of V sum
        # to zero for proportion covariances; the subtraction guards
        # inputs whose rows do not exactly sum to zero).
        tr = sum(Vm[i][i] / p0v[i] for i in range(k))
        tr -= sum(Vm[i][j] for i in range(k) for j in range(k))
        lam = tr / df
    elif deffs is not None:
        dv = [float(v) for v in deffs]
        if len(dv) != k or any(v <= 0 for v in dv):
            raise ValueError("need k positive cell deffs")
        lam = sum((1.0 - b) * d for b, d in zip(p0v, dv)) / df
    else:
        lam = 1.0
    if lam <= 0:
        raise ValueError("estimated mean design effect is not positive")
    xc = x2 / lam
    return RichResult(payload={
        "statistic": x2,
        "corrected": xc,
        "lambda_bar": lam,
        "df": df,
        "p_value": float(chi2.sf(xc, df)),
        "method": "Rao-Scott (1981) first-order corrected chi-square",
    })


# long descriptive alias (stub-era name)
rao_scott_chisq = raoscot


def cheatsheet():
    return "raoscot: X2 / mean generalized deff ~ chi2_{k-1}"

# public names resolved by fn/_lazy_map.json
raoscottchisq = raoscot
