# morie.fn -- function file (rootcoder007/morie)
"""Multiple-index model.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.2, equation (2.5) (page 10).

    E(Y | X = x) = x_0'beta_0 + G(x_1'beta_1, ..., x_M'beta_M)     (2.5)

where M is known, each x_m is a subvector of x, and G is unknown.  If
the betas are identified, they are estimable at n^{-1/2}, while the
estimator of E(Y|X=x) converges at the rate of an M-dimensional
nonparametric estimator.  So in a multiple-index model the curse of
dimensionality bites on E(Y|X=x) but not on beta (page 11).

Each beta_m is estimated up to scale by a density-weighted average
derivative within its own block (Section 2.6.1, equation (2.40)),
normalised so that its first component is one, and G is then a
product-kernel Nadaraya-Watson regression on the M fitted indices.
Bandwidths are explicit with a fixed default; there is no random
search and no cross-validation.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["multindex", "horowitz_multiple_index_model"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _avgderiv(Xb, y, h):
    """Density-weighted average derivative, eq. (2.40)."""
    n, k = Xb.shape
    W = np.ones((n, n))
    for j in range(k):
        W = W * _gauss((Xb[:, j][:, None] - Xb[:, j][None, :]) / h)
    d = np.zeros(k)
    for j in range(k):
        diff = Xb[:, j][:, None] - Xb[:, j][None, :]
        # dK/dx of the product kernel in coordinate j
        G = W * (-diff / (h * h))
        d[j] = -2.0 * float(np.sum(y[:, None] * G)) / (n * n * (h ** k))
    return d


def multindex(x, y, blocks, x0=None, h=None, hg=None, ngrid=0):
    """Fit the multiple-index model (2.5).

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    blocks : sequence of sequences of int
        Column indices of x forming each index m = 1, ..., M.
    x0 : array-like, (n, p), optional
        Covariates entering linearly (the x_0'beta_0 term).
    h : float, optional
        Bandwidth for the average-derivative step.  Default
        n**(-1/(k+4)) inside a block of width k.
    hg : float, optional
        Bandwidth for the regression of Y on the fitted indices.
        Default n**(-1/(M+4)).
    ngrid : int, default 0
        Unused placeholder for a future evaluation grid; G is returned
        at the sample indices.

    Returns
    -------
    RichResult
        payload keys: estimate, beta0, indices, ghat, resid, rss,
        betaexp, gexp, M, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    if yv.size != n:
        raise ValueError("y must have one entry per row of x.")
    M = len(blocks)
    if M < 1:
        raise ValueError("at least one index block is required.")

    resid = yv
    beta0 = None
    if x0 is not None:
        X0 = np.atleast_2d(np.asarray(x0, dtype=float))
        if X0.shape[0] != n:
            X0 = X0.T
        beta0 = np.linalg.lstsq(X0, yv, rcond=None)[0]
        resid = yv - X0 @ beta0

    betas = []
    idx = np.zeros((n, M))
    for m, cols in enumerate(blocks):
        cols = [int(j) for j in cols]
        Xb = X[:, cols]
        k = len(cols)
        hb = float(n ** (-1.0 / (k + 4.0))) if h is None else float(h)
        d = _avgderiv(Xb, resid, hb)
        if abs(float(d[0])) < 1e-300:
            raise ValueError(
                "the first covariate of an index block has a zero average "
                "derivative, so the scale normalisation beta_1 = 1 is "
                "unavailable for that block.")
        b = d / float(d[0])
        betas.append(b)
        idx[:, m] = Xb @ b

    hgv = float(n ** (-1.0 / (M + 4.0))) if hg is None else float(hg)
    W = np.ones((n, n))
    for m in range(M):
        W = W * _gauss((idx[:, m][:, None] - idx[:, m][None, :]) / hgv)
    den = np.sum(W, axis=1)
    den = np.where(den > 1e-300, den, 1e-300)
    ghat = (W @ resid) / den
    r = resid - ghat
    return RichResult(
        title="Multiple-index model (eq. 2.5)",
        payload={"estimate": betas, "beta0": beta0, "indices": idx,
                 "ghat": ghat, "resid": r,
                 "rss": float(np.sum(r * r)),
                 "betaexp": 0.5, "gexp": 2.0 / (4.0 + M),
                 "M": M, "n": n,
                 "method": "Horowitz (2009) eq. (2.5), average-derivative indices"},
    )


horowitz_multiple_index_model = multindex


def cheatsheet():
    return "hrzmim: multiple-index model E(Y|X) = x0'b0 + G(x1'b1, ..., xM'bM)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    a = np.linspace(-2, 2, n)
    b = np.cos(np.arange(n) * 0.6)
    X = np.column_stack([a, b])
    z = X @ np.array([1.0, 0.7])
    y = z + 0.3 * z ** 2
    r = multindex(X, y, [[0, 1]], h=0.6, hg=0.3)
    got = float(r["estimate"][0][1])
    assert abs(got - 0.7) < 0.25, got
    assert r["M"] == 1 and abs(r["gexp"] - 0.4) < 1e-12
    print("ok", got)
