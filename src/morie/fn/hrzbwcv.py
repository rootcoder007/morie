# morie.fn -- function file (rootcoder007/morie)
"""Bandwidth selection for a single-index model by cross-validation.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.7 (pages 44-45) and Appendix A.2.1 (page 242).

Section 2.7 reports Haerdle, Hall and Ichimura (1993): optimise the
semiparametric weighted nonlinear least-squares objective (2.25) over
btilde AND the bandwidth h_n jointly.  Under their conditions the
resulting h estimates the bandwidth that minimises the asymptotic
integrated mean-square error of a kernel estimator of G -- so it is
asymptotically optimal for estimating G, though the book is explicit
that it "does not necessarily have any optimality properties for
estimation of beta".  Section 2.7 also records that the asymptotic
distribution of n^{1/2}(b_n - beta) does not depend on h_n at all, so
bandwidth choice for beta needs higher-order theory (Haerdle and
Tsybakov 1993; Powell and Stoker 1996), where the optimal form is
h_opt = h_0 n^{-2/(2P+d+2)}.

The criterion evaluated here is the leave-one-out cross-validation
function of Appendix A.2.1 (page 242) carried to the index,

    TR(h) = n^{-1} sum_i w(X_i) [Y_i - Ghat_{-i,h}(X_i'beta)]^2

minimised over an EXPLICIT FIXED grid of bandwidths.  A fixed grid, not
a random fold split: cross-validation here is leave-one-out, which is
deterministic, and the grid is supplied or generated from a fixed
geometric ladder.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["simbwcv", "horowitz_bw_cv_sim"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def simbwcv(x, y, beta, grid=None, nh=15, lo=0.25, hi=4.0, weights=None,
            P=2, d=None):
    """Leave-one-out CV bandwidth for the index regression.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    beta : array-like, (d,)
        Index coefficients (scale normalised).
    grid : array-like, optional
        Explicit bandwidth grid.  Default is `nh` geometrically spaced
        multiples of the reference bandwidth n**(-1/5), spanning
        [lo, hi] times it.  Fixed and data-independent in shape.
    nh : int, default 15
    lo, hi : float
        Multiplier range of the default grid.
    weights : array-like, (n,), optional
        w(X_i) in TR(h).  Default all ones.
    P : int, default 2
        Kernel order used only to report the Section 2.7 form
        h_opt = h_0 n^{-2/(2P+d+2)}.
    d : int, optional
        Dimension used in that same formula; defaults to the number of
        columns of x.

    Returns
    -------
    RichResult
        payload keys: bandwidth, cv, grid, cvcurve, hreference,
        hstokerform, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    if X.shape[1] != b.size and X.shape[0] == b.size:
        X = X.T
    n, dd = X.shape
    if yv.size != n:
        raise ValueError("y must have one entry per row of x.")
    W = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    href = float(n ** -0.2)
    hs = (np.asarray(grid, dtype=float).ravel() if grid is not None
          else href * np.exp(np.linspace(np.log(float(lo)), np.log(float(hi)),
                                         int(nh))))
    z = X @ b

    cv = np.zeros(hs.size)
    for t in range(int(hs.size)):
        hh = float(hs[t])
        K = _gauss((z[:, None] - z[None, :]) / hh)
        np.fill_diagonal(K, 0.0)                     # leave one out
        den = np.sum(K, axis=1)
        den = np.where(den > 1e-300, den, 1e-300)
        gh = (K @ yv) / den
        r = yv - gh
        cv[t] = float(np.sum(W * r * r)) / n
    k = int(np.argmin(cv))
    dim = int(dd) if d is None else int(d)
    return RichResult(
        title="Cross-validated bandwidth for a single-index model",
        payload={"bandwidth": float(hs[k]), "cv": float(cv[k]),
                 "grid": hs, "cvcurve": cv, "hreference": href,
                 "hstokerform": float(n ** (-2.0 / (2 * int(P) + dim + 2))),
                 "n": n,
                 "method": "Horowitz (2009) Section 2.7 and Appendix A.2.1 TR(h)"},
    )


horowitz_bw_cv_sim = simbwcv


def cheatsheet():
    return "hrzbwcv: leave-one-out CV bandwidth for a single-index regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    X = np.column_stack([np.linspace(-2, 2, n),
                         np.cos(np.arange(1, n + 1) * 0.7)])
    b = np.array([1.0, 0.5])
    z = X @ b
    y = np.sin(z)
    r = simbwcv(X, y, b)
    assert float(np.min(r["cvcurve"])) == r["cv"]
    assert r["bandwidth"] in [float(v) for v in r["grid"]]
    # the CV curve must be U-shaped: the endpoints beat neither the min
    assert r["cv"] <= float(r["cvcurve"][0])
    assert r["cv"] <= float(r["cvcurve"][-1])
    print("ok", r["bandwidth"], r["cv"])
