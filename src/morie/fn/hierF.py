# morie.fn -- function file (rootcoder007/morie)
r"""MinT: optimal reconciliation of hierarchical forecasts.

A hierarchy of series must add up -- state forecasts should sum to the
national one -- but forecasts made independently never do. Write the
whole hierarchy as :math:`y_t = S b_t`, with :math:`S` the summing
matrix and :math:`b_t` the bottom level. Reconciliation maps incoherent
base forecasts back onto the coherent subspace,

.. math:: \tilde y_t(h) = S P\,\hat y_t(h).

**MinT chooses P by minimising the trace of the reconciled error
variance**, subject to :math:`PS = I`, and the solution is

.. math:: P = (S' W_h^{-1} S)^{-1} S' W_h^{-1},

a GLS estimator in disguise. The constraint :math:`PS = I` is what makes
:math:`SP` a projection: reconciling an already-coherent forecast leaves
it alone. That is exact, and the anchor checks it as an identity along
with :math:`(SP)(SP) = SP` -- get :math:`P` wrong and the reconciled
forecasts still add up, because :math:`S` forces that, while quietly
being the wrong point in the coherent subspace.

**The weight matrix is the whole design space, and one popular choice is
not a choice at all.** ``ols`` takes :math:`W = I`, which reconciles as
if every series had the same error variance -- rarely true when the top
of a hierarchy aggregates hundreds of noisy bottom series. ``wls``
scales by each series' own variance. ``shrink`` is the paper's
recommendation: the full covariance shrunk toward its diagonal, because
the unrestricted estimate is singular whenever there are more series
than residual observations, which in practice is usual.

**Reconciliation is not merely cosmetic.** Because it pools information
across the hierarchy, the reconciled forecast can beat the base forecast
at *every* level including the bottom -- and the anchor measures that
against a known truth rather than asserting it.

References
----------
Wickramasuriya, S. L., Athanasopoulos, G. & Hyndman, R. J. (2019)
"Optimal Forecast Reconciliation for Hierarchical and Grouped Time
Series Through Trace Minimization", *Journal of the American
Statistical Association* 114(526), 804-819,
doi:10.1080/01621459.2018.1448825. The MinT solution and its shrinkage
estimator.

Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G. & Shang, H. L. (2011)
"Optimal combination forecasts for hierarchical time series",
*Computational Statistics & Data Analysis* 55(9), 2579-2589,
doi:10.1016/j.csda.2011.03.006. The SP formulation MinT refines.

Schafer, J. & Strimmer, K. (2005) "A Shrinkage Approach to Large-Scale
Covariance Matrix Estimation and Implications for Functional Genomics",
*Statistical Applications in Genetics and Molecular Biology* 4(1),
article 32, doi:10.2202/1544-6115.1175. The shrinkage target used.

Penrose, R. (1956) "On best approximate solutions of linear matrix
equations", *Mathematical Proceedings of the Cambridge Philosophical
Society* 52(1), 17-19, doi:10.1017/S0305004100030929. The generalized
inverse the minimisation reduces to.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["summing_matrix", "mint_reconcile", "mint_P",
           "shrink_covariance", "is_coherent"]

_EPS = 1e-12


def summing_matrix(groups, n_bottom):
    r"""Build :math:`S` from a list of aggregate definitions.

    ``groups`` lists, for each aggregate row, the bottom-level indices
    it sums. The bottom level's own identity rows are appended, so
    :math:`S` is (n_agg + n_bottom) by n_bottom.
    """
    if n_bottom < 1:
        raise ValueError("hierF: need at least one bottom series")
    S = []
    for g in groups:
        row = [0.0] * n_bottom
        for i in g:
            if not 0 <= i < n_bottom:
                raise ValueError("hierF: bottom index %d out of range"
                                 % i)
            row[i] = 1.0
        S.append(row)
    for i in range(n_bottom):
        S.append([1.0 if j == i else 0.0 for j in range(n_bottom)])
    return S


def is_coherent(y, S, tol=1e-9):
    """Whether a full-hierarchy vector actually adds up."""
    m = len(S)
    n = len(S[0])
    b = y[m - n:]
    return all(abs(y[i] - sum(S[i][j] * b[j] for j in range(n))) <= tol
               for i in range(m))


def shrink_covariance(residuals, lam=None):
    r"""The paper's shrinkage estimator: the sample covariance pulled
    toward its own diagonal.

    The unrestricted covariance is singular whenever there are more
    series than residual observations, which is the usual case in a
    real hierarchy -- so the diagonal target is not a convenience, it
    is what makes the inverse exist.
    """
    T = len(residuals)
    if T < 2:
        raise ValueError("hierF: need at least 2 residual rows, got %d"
                         % T)
    m = len(residuals[0])
    mu = [sum(residuals[t][i] for t in range(T)) / T for i in range(m)]
    Sig = [[sum((residuals[t][a] - mu[a]) * (residuals[t][b] - mu[b])
                for t in range(T)) / (T - 1)
            for b in range(m)] for a in range(m)]
    D = [[Sig[a][b] if a == b else 0.0 for b in range(m)]
         for a in range(m)]
    if lam is None:
        num = sum(Sig[a][b] ** 2 for a in range(m) for b in range(m)
                  if a != b)
        # variance of the off-diagonal entries, the Schafer-Strimmer
        # intensity in its simplest form
        var = 0.0
        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                w = [(residuals[t][a] - mu[a]) * (residuals[t][b] - mu[b])
                     for t in range(T)]
                wm = sum(w) / T
                var += sum((v - wm) ** 2 for v in w) * T / (T - 1) ** 3
        lam = 1.0 if num <= _EPS else min(1.0, max(0.0, var / num))
    return [[(1.0 - lam) * Sig[a][b] + lam * D[a][b] for b in range(m)]
            for a in range(m)], lam


def mint_P(S, W=None, method="shrink", residuals=None, ridge=1e-10):
    r""":math:`P = (S'W^{-1}S)^{-1}S'W^{-1}`."""
    if method not in ("ols", "wls", "shrink", "custom"):
        raise ValueError("hierF: method must be ols, wls, shrink or "
                         "custom, got %r" % (method,))
    m = len(S)
    n = len(S[0])
    lam = None
    if method == "ols":
        Wm = [[1.0 if a == b else 0.0 for b in range(m)]
              for a in range(m)]
    elif method == "wls":
        if residuals is None:
            raise ValueError("hierF: wls needs residuals")
        T = len(residuals)
        v = [max(sum(residuals[t][a] ** 2 for t in range(T)) / T, _EPS)
             for a in range(m)]
        Wm = [[v[a] if a == b else 0.0 for b in range(m)]
              for a in range(m)]
    elif method == "shrink":
        if residuals is None:
            raise ValueError("hierF: shrink needs residuals")
        Wm, lam = shrink_covariance(residuals)
    else:
        if W is None:
            raise ValueError("hierF: method='custom' needs W")
        Wm = [list(r) for r in W]
    # solve W X = S for X = W^-1 S, one column at a time
    WX = [k.cholsolve([[Wm[a][b] + (ridge if a == b else 0.0)
                        for b in range(m)] for a in range(m)],
                      [S[a][j] for a in range(m)]) for j in range(n)]
    Winv_S = [[WX[j][a] for j in range(n)] for a in range(m)]
    A = [[sum(S[a][i] * Winv_S[a][j] for a in range(m))
          for j in range(n)] for i in range(n)]
    P = []
    for i in range(n):
        e = [1.0 if t == i else 0.0 for t in range(n)]
        row = k.cholsolve([[A[p][q] + (ridge if p == q else 0.0)
                            for q in range(n)] for p in range(n)], e)
        P.append([sum(row[j] * Winv_S[a][j] for j in range(n))
                  for a in range(m)])
    return P, lam


def mint_reconcile(base, S, method="shrink", residuals=None, W=None,
                   ridge=1e-10):
    r"""Reconcile base forecasts: :math:`\tilde y = S P \hat y`."""
    Sm = [list(r) for r in S]
    m = len(Sm)
    n = len(Sm[0])
    yb = k.vec(base)
    if len(yb) != m:
        raise ValueError("hierF: %d base forecasts for %d series"
                         % (len(yb), m))
    P, lam = mint_P(Sm, W=W, method=method, residuals=residuals,
                    ridge=ridge)
    b = [sum(P[i][a] * yb[a] for a in range(m)) for i in range(n)]
    rec = [sum(Sm[a][j] * b[j] for j in range(n)) for a in range(m)]
    # PS = I is the constraint; check it rather than trust it
    PS = [[sum(P[i][a] * Sm[a][j] for a in range(m)) for j in range(n)]
          for i in range(n)]
    ps_err = max(abs(PS[i][j] - (1.0 if i == j else 0.0))
                 for i in range(n) for j in range(n))
    return RichResult(payload={
        "estimate": rec, "reconciled": rec, "bottom": b,
        "base": list(yb), "P": P, "S": Sm, "method": method,
        "shrinkage": lam, "n_series": m, "n_bottom": n,
        "coherent": is_coherent(rec, Sm),
        "ps_identity_error": ps_err,
        "adjustment": [rec[a] - yb[a] for a in range(m)],
        "cite": "MinT, Wickramasuriya, Athanasopoulos & Hyndman (2019)",
        "method_detail": "P = (S' W^-1 S)^-1 S' W^-1",
    })


def cheatsheet():
    return ("hierF: y = S b, reconcile with ytilde = S P yhat where "
            "P = (S'W^-1 S)^-1 S'W^-1 minimises tr(P W P') subject to "
            "PS = I (MinT). PS = I makes SP a PROJECTION -- an already "
            "coherent forecast is left alone. Wrong P still adds up, "
            "because S forces that; it is just the wrong coherent "
            "point. W: ols=I, wls=diag, shrink=the paper's default "
            "because the full covariance is singular when m > T.")


# compact alias per ledger/NAMING.md
mintreconcile = mint_reconcile
