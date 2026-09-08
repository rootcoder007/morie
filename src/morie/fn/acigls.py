# morie.fn -- function file (rootcoder007/morie)
r"""Inverse-probability-weighted GLS with cluster-robust standard
errors.

**Two corrections, doing different jobs.** They are routinely
confused, so this module keeps them separate.

*The weights fix the point estimate.* When units enter the sample
with unequal probability :math:`\pi_i` -- an outcome-dependent
design, differential non-response -- the unweighted fit estimates the
sampled population, not the target one. Weighting each unit by
:math:`1/\pi_i` restores it:

.. math:: \hat\beta = (A'WA)^{-1} A'Wy, \qquad W = \mathrm{diag}(w_i).

*The cluster-robust variance fixes the standard error.* Observations
within a cluster are correlated, so the usual
:math:`\sigma^2 (A'WA)^{-1}` understates the variance -- often
badly. The sandwich sums score contributions *per cluster* rather
than per observation:

.. math:: \hat{V} = (A'WA)^{-1}
          \Bigl(\sum_{c=1}^{G} u_c u_c'\Bigr) (A'WA)^{-1},
          \qquad u_c = A_c' W_c e_c.

Weighting alone leaves the standard error wrong; the sandwich alone
leaves the coefficient wrong. Both are reported, alongside the naive
standard error, so the size of each correction is visible instead of
assumed.

**What the sandwich costs.** It is consistent as the number of
*clusters* grows, not as the number of observations grows. With few
clusters it is biased downward however many rows there are, which is
why the finite-sample factor

.. math:: \frac{G}{G-1}\cdot\frac{n-1}{n-p}

is applied and why ``n_clusters`` is returned: a cluster-robust
standard error computed from eight clusters should be read with
suspicion no matter how confident it looks.

**Degenerate cases are refused, not smoothed.** A cluster structure
with one cluster gives no between-cluster information and the
sandwich is undefined; a non-positive weight is not an
inverse-probability weight. Both raise.

References
----------
Liang, K.-Y. & Zeger, S. L. (1986) "Longitudinal data analysis using
generalized linear models", *Biometrika* 73(1), 13-22,
doi:10.1093/biomet/73.1.13. The cluster-summed sandwich reproduced
above, and its consistency in the number of clusters.

Zeger, S. L. & Liang, K.-Y. (1986) "Longitudinal data analysis for
discrete and continuous outcomes", *Biometrics* 42(1), 121-130,
doi:10.2307/2531248, for the same variance applied to continuous
responses, which is the case here.

Note on attribution: the module ledger records this entry against
Schildcrout & Heagerty (2007) "Marginalized models for moderate to
long series of longitudinal binary response data", *Biometrics*
63(2), 322-331, doi:10.1111/j.1541-0420.2006.00680.x. That paper
concerns marginalized likelihood models for *binary* longitudinal
outcomes and does not contain this estimator, so the estimator is
cited to its own source above rather than to the ledger's.
"""

import math

from ._richresult import RichResult

__all__ = ["adjusted_ipgls", "adjustedipgls"]


def _design(A, n):
    M = [[float(v) for v in row] for row in A]
    if len(M) != n:
        raise ValueError("acigls: A has %d rows but y has %d entries"
                         % (len(M), n))
    p = len(M[0]) if M else 0
    if p == 0 or any(len(r) != p for r in M):
        raise ValueError("acigls: A is ragged or has no columns")
    return M, p


def _solve(Amat, b):
    p = len(b)
    Ab = [list(Amat[i]) + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(Ab[r][c]))
        if abs(Ab[piv][c]) < 1e-12:
            raise ValueError("acigls: A'WA is singular -- the design "
                             "has collinear columns, or a weight has "
                             "removed a column's variation")
        Ab[c], Ab[piv] = Ab[piv], Ab[c]
        for r in range(p):
            if r == c:
                continue
            f = Ab[r][c] / Ab[c][c]
            for k in range(c, p + 1):
                Ab[r][k] -= f * Ab[c][k]
    return [Ab[i][p] / Ab[i][i] for i in range(p)]


def _inverse(Amat):
    p = len(Amat)
    cols = []
    for j in range(p):
        e = [1.0 if i == j else 0.0 for i in range(p)]
        cols.append(_solve(Amat, e))
    return [[cols[j][i] for j in range(p)] for i in range(p)]


def adjusted_ipgls(y, A, H, cluster, small_sample=True):
    r"""IPW-GLS with a cluster-robust variance.

    ``H`` is the inverse-probability weight of each observation.
    ``cluster`` labels the independent groups.
    """
    yy = [float(v) for v in y]
    n = len(yy)
    if n < 2:
        raise ValueError("acigls: need at least two observations")
    M, p = _design(A, n)
    w = [float(v) for v in H]
    if len(w) != n:
        raise ValueError("acigls: %d weights but %d observations"
                         % (len(w), n))
    if any(v <= 0 for v in w):
        raise ValueError("acigls: every weight must be positive; an "
                         "inverse-probability weight cannot be zero "
                         "or negative")
    cl = [str(v) for v in cluster]
    if len(cl) != n:
        raise ValueError("acigls: %d cluster labels but %d "
                         "observations" % (len(cl), n))
    groups = {}
    for i, c in enumerate(cl):
        groups.setdefault(c, []).append(i)
    G = len(groups)
    if G < 2:
        raise ValueError("acigls: the cluster-robust variance needs "
                         "at least two clusters; with one there is no "
                         "between-cluster information")
    if n <= p:
        raise ValueError("acigls: %d observations cannot support %d "
                         "coefficients" % (n, p))

    XtWX = [[sum(w[i] * M[i][a] * M[i][b] for i in range(n))
             for b in range(p)] for a in range(p)]
    XtWy = [sum(w[i] * M[i][a] * yy[i] for i in range(n))
            for a in range(p)]
    beta = _solve(XtWX, XtWy)
    resid = [yy[i] - sum(M[i][k] * beta[k] for k in range(p))
             for i in range(n)]

    bread = _inverse(XtWX)
    meat = [[0.0] * p for _ in range(p)]
    for _c, idx in groups.items():
        u = [sum(w[i] * M[i][a] * resid[i] for i in idx)
             for a in range(p)]
        for a in range(p):
            for b in range(p):
                meat[a][b] += u[a] * u[b]
    corr = ((G / float(G - 1)) * ((n - 1) / float(n - p))
            if small_sample else 1.0)
    V = [[corr * sum(bread[a][k] * meat[k][l] * bread[l][b]
                     for k in range(p) for l in range(p))
          for b in range(p)] for a in range(p)]
    se = [math.sqrt(V[k][k]) if V[k][k] > 0 else float("nan")
          for k in range(p)]

    sw = sum(w)
    s2 = sum(w[i] * resid[i] ** 2 for i in range(n)) / (n - p)
    naive = [math.sqrt(s2 * bread[k][k]) if bread[k][k] > 0
             else float("nan") for k in range(p)]
    return RichResult(payload={
        "estimate": beta, "coefficients": beta,
        "std_errors": se, "naive_std_errors": naive,
        "vcov": V, "residuals": resid,
        "n": n, "n_clusters": G, "n_coefficients": p,
        "sum_weights": sw,
        "finite_sample_correction": corr,
        "inflation": [se[k] / naive[k] if naive[k] > 0 else float("nan")
                      for k in range(p)],
        "method": "IPW-GLS with a cluster-robust sandwich variance "
                  "(Liang & Zeger 1986)",
    })


def cheatsheet():
    return ("acigls: beta = (A'WA)^-1 A'Wy with W = diag(1/pi), and a "
            "variance that sums scores PER CLUSTER, not per row. The "
            "weights fix the estimate; the sandwich fixes the standard "
            "error; neither substitutes for the other. Consistent in "
            "the number of CLUSTERS, so few clusters means a "
            "downward-biased SE however many rows there are.")


# compact alias per ledger/NAMING.md
adjustedipgls = adjusted_ipgls
