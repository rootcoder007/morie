# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric additive model by marginal integration.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 3.1.1, equations (3.6)-(3.9) (pages 55-57).

Model (3.5):  E(Y | X = x) = mu + m_1(x^1) + ... + m_d(x^d).

The decomposition is only unique up to constants shifted between the
components and mu, so marginal integration identifies it with the
location normalisation

    E[m_j(X^j)] = 0;  j = 1, ..., d                            (3.6)

Then E(Y) = mu (3.7), and integrating the full regression over the
marginal density of the other covariates,

    m_1(x^1) = integral E(Y|X=x) p_{-1}(x^{-1}) dx^{-1} - mu   (3.8)

The sample analogs are muhat = (1/n) sum_i Y_i and, with the product
kernel estimator ghat of E(Y|X) of (3.9), the average of
ghat(x^1, X_i^{-1}) over i.  Bandwidths h1 and h2 are explicit with a
fixed default; nothing is cross-validated and nothing is random.

The additive structure is what buys the escape from the curse of
dimensionality: each m_j is a one-dimensional object.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["npaddreg", "horowitz_additive_model"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def npaddreg(x, y, h1=None, h2=None, ngrid=25, grids=None):
    """Estimate mu and the additive components by marginal integration.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    h1 : float, optional
        Bandwidth for the direction being estimated.  Default
        n**(-1/5).
    h2 : float, optional
        Bandwidth for the remaining d-1 directions.  Default
        n**(-1/(d+3)).
    ngrid : int, default 25
        Points per component grid.
    grids : sequence of array-like, optional
        Explicit evaluation grids, one per component.

    Returns
    -------
    RichResult
        payload keys: mu, grids, components, fitted, resid, rss, h1,
        h2, d, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("an additive model needs at least two covariates.")
    a1 = float(n ** -0.2) if h1 is None else float(h1)
    a2 = float(n ** (-1.0 / (d + 3.0))) if h2 is None else float(h2)
    mu = float(np.mean(yv))                                     # (3.7)

    gs = []
    comps = []
    for j in range(d):
        g = (np.linspace(float(np.min(X[:, j])), float(np.max(X[:, j])),
                         int(ngrid))
             if grids is None else
             np.atleast_1d(np.asarray(grids[j], dtype=float)))
        gs.append(g)
        # product kernel weights in the other directions, evaluated at
        # each sample point: W2[i, l] = prod_{k != j} K((X_i^k - X_l^k)/h2)
        W2 = np.ones((n, n))
        for k in range(d):
            if k == j:
                continue
            W2 = W2 * _gauss((X[:, k][:, None] - X[:, k][None, :]) / a2)
        # kernel in direction j from the grid to the sample
        K1 = _gauss((g[:, None] - X[:, j][None, :]) / a1)
        mj = np.zeros(g.size)
        for t in range(int(g.size)):
            Wt = W2 * K1[t][None, :]                            # (3.9)
            den = np.sum(Wt, axis=1)
            den = np.where(den > 1e-300, den, 1e-300)
            ghat = (Wt @ yv) / den
            mj[t] = float(np.mean(ghat))                        # (3.8)
        comps.append(mj - float(np.mean(mj)))                   # (3.6)

    fit = np.full(n, mu)
    for j in range(d):
        fit = fit + np.interp(X[:, j], gs[j], comps[j])
    r = yv - fit
    return RichResult(
        title="Nonparametric additive model by marginal integration",
        payload={"mu": mu, "grids": gs, "components": comps,
                 "fitted": fit, "resid": r, "rss": float(np.sum(r * r)),
                 "h1": a1, "h2": a2, "d": d, "n": n,
                 "method": "Horowitz (2009) eq. (3.6)-(3.9) marginal integration"},
    )


horowitz_additive_model = npaddreg


def cheatsheet():
    return "hrzadm: nonparametric additive model by marginal integration (eq. 3.8)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 150
    x1 = np.linspace(-2, 2, n)
    x2 = np.cos(np.arange(1, n + 1) * 0.9) * 2.0
    y = 1.0 + x1 + 0.5 * x2 ** 2
    r = npaddreg(np.column_stack([x1, x2]), y, h1=0.4, h2=0.5)
    assert abs(r["mu"] - float(np.mean(y))) < 1e-12
    # component 1 must be increasing in x1 and centred
    c1 = r["components"][0]
    assert float(np.mean(c1)) < 1e-12
    assert float(c1[-1]) > float(c1[0])
    print("ok", r["mu"], float(c1[0]), float(c1[-1]))
