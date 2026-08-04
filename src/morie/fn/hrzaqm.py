# morie.fn -- function file (rootcoder007/morie)
"""Additive model of a conditional quantile function.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 3.4, equation (3.29) (pages 81-82).

    Y = mu + m_1(x^1) + ... + m_d(x^d) + U_alpha                (3.29)

where the alpha-quantile of U_alpha conditional on X = x is zero for
almost every x.  The book describes the two-stage estimator of
Horowitz and Lee (2005): a series approximation fitted by minimising

    rho_alpha(u) = |u| + (2 alpha - 1) u

over the coefficients in stage one -- the same shape as the mean case
but with the check loss in place of squared error -- then a local
second stage, giving an estimator that is asymptotically normal,
oracle efficient, and free of the curse of dimensionality.  The
location normalisation Horowitz and Lee use is
integral_{-1}^{1} m_j(v) dv = 0 on the support [-1, 1]^d.

Here stage one is a polynomial series fitted by a FIXED number of IRLS
iterations on the check loss (no tolerance-based early exit), and
stage two is a kernel-weighted check-loss fit of the working residual
in each direction.  Components are centred over their grids, which is
the discrete form of the book's integral normalisation.
"""

from __future__ import annotations

from . import _array_core as np
from ._horowitz import qirls

from ._richresult import RichResult

__all__ = ["addquant", "horowitz_additive_quantile"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def addquant(x, y, alpha=0.5, K=4, h=None, niter=40, ngrid=25):
    """Additive conditional-quantile model (3.29).

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    alpha : float, default 0.5
        Quantile level in (0, 1).
    K : int, default 4
        Series length per coordinate.
    h : float, optional
        Second-stage bandwidth.  Default n**(-1/5).
    niter : int, default 40
        FIXED IRLS iterations; no tolerance-based early exit.
    ngrid : int, default 25

    Returns
    -------
    RichResult
        payload keys: mu, grids, components, fitted, resid, checkloss,
        alpha, bandwidth, K, d, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("an additive model needs at least two covariates.")
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    hh = float(n ** -0.2) if h is None else float(h)
    Ki = int(K)

    cols = [np.ones(n)]
    for j in range(d):
        sc = X[:, j]
        rng = float(np.max(sc) - np.min(sc))
        sc = (sc - float(np.min(sc))) / (rng if rng > 0 else 1.0) * 2.0 - 1.0
        for k in range(1, Ki + 1):
            cols.append(sc ** k)
    P = np.column_stack(cols)
    theta = qirls(P, yv, np.ones(n), a, niter=int(niter))
    stage1 = P @ theta

    gs = []
    comps = []
    resid1 = yv - stage1
    for j in range(d):
        g = np.linspace(float(np.min(X[:, j])), float(np.max(X[:, j])),
                        int(ngrid))
        gs.append(g)
        mj = np.zeros(g.size)
        for t in range(int(g.size)):
            w = _gauss((float(g[t]) - X[:, j]) / hh)
            Dm = np.column_stack([np.ones(n), X[:, j] - float(g[t])])
            co = qirls(Dm, resid1, w, a, niter=int(niter))
            mj[t] = float(co[0])
        comps.append(mj - float(np.mean(mj)))

    mu = float(np.mean(stage1))
    fit = np.full(n, mu) + (stage1 - mu)
    for j in range(d):
        fit = fit + np.interp(X[:, j], gs[j], comps[j])
    r = yv - fit
    loss = float(np.sum(np.abs(r) + (2.0 * a - 1.0) * r))
    return RichResult(
        title="Additive conditional-quantile model (eq. 3.29)",
        payload={"mu": mu, "grids": gs, "components": comps,
                 "fitted": fit, "resid": r, "checkloss": loss,
                 "alpha": a, "bandwidth": hh, "K": Ki, "d": d, "n": n,
                 "method": "Horowitz (2009) eq. (3.29), check-loss series then local fit"},
    )


horowitz_additive_quantile = addquant


def cheatsheet():
    return "hrzaqm: additive conditional-quantile model via the check loss (eq. 3.29)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 120
    x1 = np.linspace(-2, 2, n)
    x2 = np.cos(np.arange(1, n + 1) * 0.9)
    y = 1.0 + 0.7 * x1 + 0.4 * x2 ** 2
    r = addquant(np.column_stack([x1, x2]), y, alpha=0.5, h=0.5)
    assert r["checkloss"] >= 0.0
    # the additive median must track the additive mean here
    assert float(np.max(np.abs(r["fitted"] - y))) < 1e-6, r["checkloss"]
    # each component is centred over its grid (the discrete form of the
    # Horowitz-Lee integral normalisation)
    for c in r["components"]:
        assert abs(float(np.mean(c))) < 1e-10
    # a genuinely nonpolynomial shape leaves nonzero components
    y2 = 1.0 + np.abs(x1) + 0.4 * x2
    r2 = addquant(np.column_stack([x1, x2]), y2, alpha=0.5, h=0.5)
    assert float(np.max(np.abs(r2["components"][0]))) > 1e-6
    print("ok", r["checkloss"], float(np.max(np.abs(r2["components"][0]))))
