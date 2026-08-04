# morie.fn -- slice s04 (rootcoder007/morie)
"""CV1 genomic cross-validation: train on observed, predict unobserved lines.

Book sections read: Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*, Springer -- volume [Pages 109-139], Chapter 4.  Section
4.3.6, p. 118, defines the scheme: the lines are partitioned into g
groups, "the information of g-1 groups are used as the training set while
all individuals of the remaining group are used as the testing set.
... Jarquin et al. (2017) denotes this type of CV strategy as CV1".
Section 4.5.1, equation (4.2), p. 129, gives the predictive ability as
Pearson's correlation between the testing observations and their
predictions, and equation (4.1) gives the testing MSE.

Section 5.3 of volume [Pages 141-170], equation (5.3), supplies the
predictor: the GBLUP mean 1_n mu + Z_L b with b ~ N(0, sigma_g^2 G).
Written on the marker scale that is the ridge predictor used here.

DETERMINISM.  The folds are not drawn.  Line i goes to fold i mod K, the
in-order complementary partition, so that K = n is exactly leave-one-out
and both arms partition identically.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["cv1_genomic"]


def cv1_genomic(y, markers, n_folds, lam=1.0):
    """CV1 predictive ability of a ridge (GBLUP) predictor.

    Parameters
    ----------
    y : array-like
        The n phenotypes.
    markers : array-like
        n-by-p marker matrix.
    n_folds : int
        Number of complementary groups, 2 <= K <= n.
    lam : float
        Ridge penalty standing for sigma^2/sigma_beta^2.

    Returns
    -------
    estimate : pa, the overall Pearson correlation of eq. (4.2)
    pa       : the same value
    mse      : the overall testing MSE of eq. (4.1)
    y_hat    : the out-of-fold predictions, in the order of y
    pa_fold  : the per-fold correlations
    fold     : the fold label of each line
    """
    yy = core.vec(y)
    n = len(yy)
    if n < 2:
        raise ValueError("cv1_genomic: need at least two lines")
    X = core.mat(markers)
    if len(X) != n:
        raise ValueError("cv1_genomic: markers has a different number of rows than y")
    p = len(X[0])
    K = int(n_folds)
    if K < 2 or K > n:
        raise ValueError("cv1_genomic: n_folds must lie between 2 and the number of lines")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("cv1_genomic: lam must be non-negative")
    fold = [i % K for i in range(n)]
    yhat = [0.0] * n
    for f in range(K):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        if not tr or not te:
            raise ValueError("cv1_genomic: a fold left no training or no testing lines")
        mu = 0.0
        for i in tr:
            mu += yy[i]
        mu = mu / len(tr)
        A = [[0.0] * p for _ in range(p)]
        r = [0.0] * p
        for i in tr:
            d = yy[i] - mu
            for a in range(p):
                r[a] += X[i][a] * d
                for c in range(p):
                    A[a][c] += X[i][a] * X[i][c]
        for a in range(p):
            A[a][a] += lam
        beta = core.ridgesolve(A, r, 1e-12)
        for i in te:
            s = mu
            for a in range(p):
                s += X[i][a] * beta[a]
            yhat[i] = s
    pa_fold = []
    for f in range(K):
        te = [i for i in range(n) if fold[i] == f]
        pa_fold.append(core.corr([yy[i] for i in te], [yhat[i] for i in te])
                       if len(te) > 1 else float("nan"))
    s = 0.0
    for i in range(n):
        d = yy[i] - yhat[i]
        s += d * d
    return RichResult(
        title="CV1 genomic cross-validation",
        summary_lines=[("lines", n), ("markers", p), ("folds", K)],
        payload={
            "estimate": core.corr(yy, yhat),
            "pa": core.corr(yy, yhat),
            "mse": s / n,
            "y_hat": yhat,
            "pa_fold": pa_fold,
            "fold": fold,
            "n": n,
            "method": "CV1 of Chapter 4 Sect. 4.3.6 scored by eqs. (4.1)-(4.2), GBLUP predictor of eq. (5.3)",
        },
    )


def cheatsheet():
    return "cv1gn: CV1 genomic cross-validation: train on observed, predict unobserved lines"


# compact alias per ledger/NAMING.md
cv1genomic = cv1_genomic
