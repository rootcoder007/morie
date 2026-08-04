# morie.fn -- function file (rootcoder007/morie)
"""Single-index model of a conditional quantile function.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.9, equations (2.56)-(2.57) (pages 48-49).

    Q_alpha(Y | X = x) = G(x'beta)                              (2.56)

G and beta are identified under the assumptions of Theorem 2.1, and if
b_n is n^{-1/2}-consistent for beta then G follows from a
one-dimensional nonparametric quantile regression of Y on X'b_n.  As
in the conditional-mean case, beta is proportional to
dQ_alpha(Y|X=x)/dx, so with a weight function W,

    delta = E[ W(x) dQ_alpha(Y|X = x)/dx ]                      (2.57)

equals beta up to a proportionality constant, and the average
derivative estimator is

    deltahat_AD = (1/n) sum_i [dQhat_alpha(Y|X_i)/dx] W(X_i)

(Chaudhuri, Doksum and Samarov 1997 derive its asymptotics.)

The local derivative is taken from a kernel-weighted LINEAR quantile
regression fitted at each X_i by a FIXED number of IRLS iterations on
the check loss -- no tolerance-based early exit -- and the resulting
delta is rescaled so its first component is one.
"""

from __future__ import annotations

from . import _array_core as np
from ._horowitz import qirls

from ._richresult import RichResult

__all__ = ["simquant", "horowitz_sim_quantile"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def simquant(x, y, alpha=0.5, h=None, hg=None, niter=40, ngrid=25,
             weights=None):
    """Average-derivative estimator of beta in the quantile index model.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    alpha : float, default 0.5
    h : float, optional
        Bandwidth of the local quantile fits.  Default n**(-1/(d+4)).
    hg : float, optional
        Bandwidth for the quantile regression of Y on the fitted
        index.  Default n**(-1/5).
    niter : int, default 40
        FIXED IRLS iterations.
    ngrid : int, default 25
    weights : array-like, (n,), optional
        W(X_i) in (2.57).  Default all ones.

    Returns
    -------
    RichResult
        payload keys: estimate, delta, index, grid, ghat, alpha,
        bandwidth, hg, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("a single-index model needs at least two covariates.")
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    hh = float(n ** (-1.0 / (d + 4.0))) if h is None else float(h)
    hgv = float(n ** -0.2) if hg is None else float(hg)
    W = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()

    delta = np.zeros(d)
    for i in range(n):
        u = (X - X[i][None, :]) / hh
        w = np.ones(n)
        for j in range(d):
            w = w * _gauss(u[:, j])
        Dm = np.column_stack([np.ones(n), X - X[i][None, :]])
        co = qirls(Dm, yv, w, a, niter=int(niter))
        delta = delta + W[i] * co[1:]
    delta = delta / n                                            # (2.57)
    if abs(float(delta[0])) < 1e-300:
        raise ValueError(
            "the average derivative in the first coordinate is zero, so the "
            "scale normalisation beta_1 = 1 is unavailable.")
    beta = delta / float(delta[0])

    z = X @ beta
    g = np.linspace(float(np.min(z)), float(np.max(z)), int(ngrid))
    gh = np.zeros(g.size)
    for t in range(int(g.size)):
        w = _gauss((float(g[t]) - z) / hgv)
        Dm = np.column_stack([np.ones(n), z - float(g[t])])
        co = qirls(Dm, yv, w, a, niter=int(niter))
        gh[t] = float(co[0])
    return RichResult(
        title="Single-index conditional-quantile model (eq. 2.56)",
        payload={"estimate": beta, "delta": delta, "index": z,
                 "grid": g, "ghat": gh, "alpha": a, "bandwidth": hh,
                 "hg": hgv, "n": n,
                 "method": "Horowitz (2009) eq. (2.56)-(2.57) quantile average derivative"},
    )


horowitz_sim_quantile = simquant


def cheatsheet():
    return "hrzsiqm: single-index conditional-quantile model, average derivative"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 90
    X = np.column_stack([np.linspace(-2, 2, n),
                         np.cos(np.arange(1, n + 1) * 0.9)])
    z = X @ np.array([1.0, 0.7])
    y = z + 0.2 * z ** 2                       # median = the index itself
    r = simquant(X, y, alpha=0.5, h=0.8, hg=0.4)
    got = float(r["estimate"][1])
    assert abs(got - 0.7) < 0.3, got
    assert abs(float(r["estimate"][0]) - 1.0) < 1e-12
    print("ok", got)
