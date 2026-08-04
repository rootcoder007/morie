# morie.fn -- function file (rootcoder007/morie)
"""Identification of beta in the partially linear model.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 3.6.1, equations (3.30)-(3.33) (pages 85-86).

Model (3.2a):  Y = X'beta + g(Z) + U,  E(U | X, Z) = 0.

Taking conditional expectations given Z and differencing,

    Y - E(Y|Z) = [X - E(X|Z)]'beta + U                        (3.32)

so with Xtilde = X - E(X|Z),

    beta = [E(Xtilde Xtilde')]^{-1} E(Xtilde Ytilde)

provided the matrix

    Sigma_X = E[X - E(X|Z)][X - E(X|Z)]' > 0                   (3.33)

is positive definite.  (3.33) fails whenever X is a deterministic
function of Z, and it rules out an intercept in X, because any
intercept is absorbed into g.

This function reports Sigma_X and its eigenvalues.  E(X|Z) is taken
from a Nadaraya-Watson fit on a product Gaussian kernel with an
explicit bandwidth, so nothing here is random or tuned by a
cross-validated fold split.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["plrident", "horowitz_plr_identification"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _condmean(Z, V, h):
    """E(V | Z = Z_i) by Nadaraya-Watson with a product Gaussian kernel."""
    n, q = Z.shape
    W = np.ones((n, n))
    for j in range(q):
        W = W * _gauss((Z[:, j][:, None] - Z[:, j][None, :]) / h)
    den = np.sum(W, axis=1)
    den = np.where(den > 1e-300, den, 1e-300)
    return (W @ V) / den[:, None]


def plrident(x, z, h=None, tol=1e-10):
    """Check condition (3.33) for identification of beta.

    Parameters
    ----------
    x : array-like, (n, p)
        Covariates entering linearly.
    z : array-like, (n, q)
        Covariates entering through the unknown g.
    h : float, optional
        Kernel bandwidth.  Default is the fixed formula
        h = n**(-1/(4+q)), which is deterministic given (n, q); it is
        not chosen by cross-validation.
    tol : float, default 1e-10
        Eigenvalues at or below tol*max(eig) count as zero.

    Returns
    -------
    RichResult
        payload keys: identified, mineig, maxeig, condnum, eigvals,
        rank, dim, hasintercept, bandwidth, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != X.shape[0]:
        Z = Z.T
    n, p = X.shape
    q = Z.shape[1]
    hh = float(n ** (-1.0 / (4.0 + q))) if h is None else float(h)

    hasintercept = bool(any(float(np.std(X[:, j])) <= 0.0 for j in range(p)))
    Xt = X - _condmean(Z, X, hh)
    Sigma = (Xt.T @ Xt) / float(n)
    ev = np.linalg.eigvalsh(Sigma)
    ev = np.sort(ev)
    mineig = float(ev[0])
    maxeig = float(ev[-1])
    rank = int(np.sum((ev > maxeig * tol).astype(float)))
    condnum = float(maxeig / mineig) if mineig > 0 else float("inf")
    identified = bool(mineig > maxeig * tol and not hasintercept)
    return RichResult(
        title="Partially linear model identification (eq. 3.33)",
        payload={"identified": identified, "mineig": mineig,
                 "maxeig": maxeig, "condnum": condnum, "eigvals": ev,
                 "rank": rank, "dim": p, "hasintercept": hasintercept,
                 "bandwidth": hh, "n": n,
                 "method": "Horowitz (2009) eq. (3.33), Sigma_X positive definite"},
    )


horowitz_plr_identification = plrident


def cheatsheet():
    return "hrzplri: partially linear model identification (eq. 3.33)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 120
    z = np.linspace(-2, 2, n)
    # X varies independently of Z -> Sigma_X positive definite
    x1 = np.cos(np.arange(n) * 1.7)
    x2 = np.sin(np.arange(n) * 0.9)
    r = plrident(np.column_stack([x1, x2]), z.reshape(-1, 1))
    assert r["identified"], r
    # X a deterministic function of Z -> (3.33) fails
    bad = plrident(np.column_stack([z, z ** 2]), z.reshape(-1, 1))
    assert bad["mineig"] < r["mineig"], (bad["mineig"], r["mineig"])
    print("ok", r["mineig"], bad["mineig"])
