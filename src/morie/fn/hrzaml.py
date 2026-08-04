# morie.fn -- function file (rootcoder007/morie)
"""Additive model with a known nonidentity link function.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 3.2, equation (3.19) (pages 70-71).

    E(Y | X = x) = G[mu + m_1(x^1) + ... + m_d(x^d)]           (3.19)

with G KNOWN and not necessarily the identity.  The book's two-stage
recipe is: impose the additive structure through a series
approximation and fit its coefficients by nonlinear least squares --
which is what avoids the curse of dimensionality -- then take ONE
Newton step from the first-stage estimate toward a local-linear or
local-constant estimate.  In large samples the second-stage estimator
of each component behaves like a one-dimensional smoother with the
other components known, so it is oracle efficient and asymptotically
normal.

Here the first stage is a polynomial series in each coordinate fitted
by a FIXED number of Gauss-Newton iterations on the nonlinear least
squares criterion -- no tolerance-based early exit -- and the second
stage is the single Newton step the book prescribes, taken toward a
local-constant fit of the working residual in each direction.

Location normalisation: each component is centred, as in (3.6).
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["addlink", "horowitz_additive_nonid_link"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def _logistic(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500.0, 500.0)))


def _dlogistic(v):
    p = _logistic(v)
    return p * (1.0 - p)


_LINKS = {
    "identity": (lambda v: v, lambda v: np.ones_like(v)),
    "logistic": (_logistic, _dlogistic),
    "exp": (lambda v: np.exp(np.clip(v, -500.0, 500.0)),
            lambda v: np.exp(np.clip(v, -500.0, 500.0))),
}


def addlink(x, y, link="logistic", K=4, h=None, niter=20, ngrid=25):
    """Two-stage estimator of (3.19) with a known link.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    link : {"identity", "logistic", "exp"} or (G, Gprime) pair
    K : int, default 4
        Series length per coordinate (polynomial degree).
    h : float, optional
        Bandwidth of the Newton step.  Default n**(-1/5).
    niter : int, default 20
        FIXED Gauss-Newton iterations of the first stage; there is no
        tolerance-based early exit.
    ngrid : int, default 25

    Returns
    -------
    RichResult
        payload keys: mu, grids, components, fitted, eta, resid, rss,
        bandwidth, K, d, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("an additive model needs at least two covariates.")
    if isinstance(link, str):
        if link not in _LINKS:
            raise ValueError(
                "link must be one of %s or a (G, Gprime) pair, got %r."
                % (sorted(_LINKS), link))
        G, Gp = _LINKS[link]
    else:
        G, Gp = link
    hh = float(n ** -0.2) if h is None else float(h)
    Ki = int(K)

    # series design: intercept plus powers 1..K in each coordinate
    cols = [np.ones(n)]
    for j in range(d):
        sc = X[:, j]
        rng = float(np.max(sc) - np.min(sc))
        sc = (sc - float(np.min(sc))) / (rng if rng > 0 else 1.0) * 2.0 - 1.0
        for k in range(1, Ki + 1):
            cols.append(sc ** k)
    P = np.column_stack(cols)
    theta = np.zeros(P.shape[1])
    theta[0] = float(np.mean(yv))
    for _ in range(int(niter)):                     # FIXED iterations
        eta = P @ theta
        w = Gp(eta)
        r = yv - G(eta)
        A = P.T @ (P * (w * w)[:, None]) + 1e-8 * np.eye(P.shape[1])
        b = P.T @ (w * r)
        theta = theta + np.linalg.solve(A, b)
    eta = P @ theta

    # one Newton step toward a local-constant fit, per direction
    gs = []
    comps = []
    w = Gp(eta)
    work = eta + np.where(np.abs(w) > 1e-12, (yv - G(eta)) / np.where(
        np.abs(w) > 1e-12, w, 1.0), 0.0)
    for j in range(d):
        g = np.linspace(float(np.min(X[:, j])), float(np.max(X[:, j])),
                        int(ngrid))
        gs.append(g)
        Kj = _gauss((g[:, None] - X[:, j][None, :]) / hh)
        den = np.sum(Kj, axis=1)
        den = np.where(den > 1e-300, den, 1e-300)
        mj = (Kj @ work) / den
        comps.append(mj - float(np.mean(mj)))

    mu = float(np.mean(work))
    fit_eta = np.full(n, mu)
    for j in range(d):
        fit_eta = fit_eta + np.interp(X[:, j], gs[j], comps[j])
    fitted = G(fit_eta)
    r = yv - fitted
    return RichResult(
        title="Additive model with a known nonidentity link (eq. 3.19)",
        payload={"mu": mu, "grids": gs, "components": comps,
                 "fitted": fitted, "eta": fit_eta, "resid": r,
                 "rss": float(np.sum(r * r)), "bandwidth": hh,
                 "K": Ki, "d": d, "n": n,
                 "method": "Horowitz (2009) eq. (3.19), series then one Newton step"},
    )


horowitz_additive_nonid_link = addlink


def cheatsheet():
    return "hrzaml: additive model with a known nonidentity link (eq. 3.19)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 150
    x1 = np.linspace(-2, 2, n)
    x2 = np.cos(np.arange(1, n + 1) * 0.9)
    eta = 0.5 * x1 + 0.8 * x2
    y = 1.0 / (1.0 + np.exp(-eta))
    r = addlink(np.column_stack([x1, x2]), y, link="logistic", h=0.4)
    assert r["rss"] < 0.5, r["rss"]
    c1 = r["components"][0]
    assert float(c1[-1]) > float(c1[0])       # increasing in x1
    print("ok", r["rss"], float(c1[0]), float(c1[-1]))
