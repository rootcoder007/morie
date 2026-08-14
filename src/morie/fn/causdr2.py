# SPDX-License-Identifier: AGPL-3.0-or-later
"""DML2 partially linear regression with the Neyman-orthogonal score."""

from . import _array_core as np

from ._richresult import RichResult
from ._rrng_core import RRandom

__all__ = ["causdr2", "causal_orthogonal_score"]


def _ols_fit(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def causdr2(y, d, X, K=2, seed=1):
    """
    Double/debiased machine learning for the partially linear model
    Y = theta D + g(X) + U with the Robinson partialling-out score and
    K-fold cross-fitting (DML2), using OLS nuisance learners.

    The Neyman-orthogonal score is
    psi(W; theta, eta) = (Y - l(X) - theta (D - m(X))) (D - m(X)) with
    l(X) = E[Y | X] and m(X) = E[D | X] (Chernozhukov et al. 2018,
    Eq. 4.4). DML2 (their Definition 3.2) fits the nuisances on each
    fold complement, then solves the pooled empirical moment
    sum_i psi(W_i; theta, eta_k(i)) = 0, giving

        theta = sum(Vhat (Y - lhat)) / sum(Vhat (D - mhat)),
        Vhat = D - mhat.

    The variance estimator is the plug-in from their Theorem 3.2:
    sigma2 = J^-2 mean(psi^2) with J = mean(-Vhat^2), reported as
    se = sqrt(sigma2 / n). The K-fold partition is a random split by
    the R-compatible Mersenne-Twister stream (seeded shuffle, folds of
    near-equal size), so both language arms produce identical folds.

    Parameters
    ----------
    y : array-like
        Outcome, length n.
    d : array-like
        Treatment (continuous or binary).
    X : array-like, shape (n, p)
        Controls.
    K : int
        Number of cross-fitting folds (default 2). K = 1 disables
        cross-fitting (nuisances fit on the full sample; useful for
        closed-form anchoring, not recommended for inference).
    seed : int
        Seed for the fold shuffle.

    Returns
    -------
    result : RichResult
        Keys: estimate (theta), se, K, n, folds (fold id per unit).

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
    C., Newey, W. and Robins, J. (2018), "Double/debiased machine
    learning for treatment and structural parameters", The
    Econometrics Journal 21(1), C1-C68, doi:10.1111/ectj.12097;
    Eq. 4.4 (partialling-out score), Definition 3.2 (DML2),
    Theorem 3.2 (variance). Local copy (arXiv:1608.00060):
    fetched-wave3/chernozhukov-etal-2018-double-debiased-machine-learning-EJ21.pdf
    """
    yv = np.asarray(y, dtype=float)
    dv = np.asarray(d, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    n = len(yv)
    if Xa.shape[0] != n or len(dv) != n:
        raise ValueError("y, d, X must have matching first dimension")
    K = int(K)
    if K < 1 or K > n:
        raise ValueError("K must lie in 1..n")
    # The nuisance regressions add their own intercept, so a constant
    # column in X makes the design singular. That surfaced as a bare
    # "singular matrix" from the linear algebra core, which says nothing
    # about which argument was wrong.
    Xl = Xa.tolist()
    for j in range(len(Xl[0])):
        col = [row[j] for row in Xl]
        if max(col) - min(col) == 0.0:
            raise ValueError(
                "causdr2: column %d of X is constant; the nuisance "
                "regressions add their own intercept, so do not pass "
                "one" % j)
    Dg = np.concatenate([np.ones((n, 1)), Xa], axis=1)
    if K == 1:
        folds = [0] * n
    else:
        rng = RRandom(seed)
        perm = [i - 1 for i in rng.sample_int(n)]
        folds = [0] * n
        for pos, i in enumerate(perm):
            folds[i] = pos % K
    lhat = np.zeros(n)
    mhat = np.zeros(n)
    for k in range(K):
        tr = [i for i in range(n) if folds[i] != k] if K > 1 else list(range(n))
        te = [i for i in range(n) if folds[i] == k]
        Dtr = np.stack([Dg[i] for i in tr], axis=0)
        bl = _ols_fit(Dtr, np.asarray([yv[i] for i in tr]))
        bm = _ols_fit(Dtr, np.asarray([dv[i] for i in tr]))
        for i in te:
            lhat[i] = float(Dg[i] @ bl)
            mhat[i] = float(Dg[i] @ bm)
    v = dv - mhat
    ry = yv - lhat
    denom = float(v @ v)
    if denom == 0.0:
        raise ValueError("treatment residual is identically zero")
    theta = float(v @ ry) / denom
    psi = (ry - theta * v) * v
    J = -denom / n
    sigma2 = float(np.mean(psi * psi)) / (J * J)
    return RichResult(payload={
        "estimate": theta,
        "se": float(np.sqrt(sigma2 / n)),
        "K": K, "n": n,
        "folds": [f + 1 for f in folds],
        "method": "Chernozhukov et al. (2018) DML2, partialling-out score Eq. 4.4",
    })


causal_orthogonal_score = causdr2


def cheatsheet():
    return "causdr2(y, d, X, K, seed) -> DML2 partially linear theta with cross-fit OLS nuisances."

# public names resolved by fn/_lazy_map.json
causal_dr_orthogonal = causdr2
