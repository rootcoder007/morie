# morie.fn -- function file (rootcoder007/morie)
"""Pointwise confidence bands for a nonparametric mean regression.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Appendix A.2.1, pages 240-241.

The Nadaraya-Watson estimator is

    g_n(x) = (1 / (n h_n p_n(x))) sum_i Y_i K((x - X_i) / h_n)

and with h_n = c n^{-1/(2s+1)}, an order s kernel and s continuous
derivatives of p and g,

    n^{s/(2s+1)} [g_n(x) - g(x)] -> N(mu_R, sigma_R^2),
    sigma_R^2 = B sigma^2(x) / (c p(x)),   B = integral K(v)^2 dv.

Pointwise bands follow directly:

    g_n(x) +/- z_{1-alpha/2} sqrt(B sigmahat^2(x) / (n h p_n(x))).

The bias mu_R is NOT subtracted.  The book gives mu_R = c^s A D(x) /
p(x) in terms of the unknown derivatives of g and p and does not give
an estimator for it, so the bands here are bias-uncorrected and are
honest about it; the book's own remedy is undersmoothing or the
bias-removal method of Schucany and Sommers (1977) that it cites.

The book states the UNIFORM rate sup_x |g_n(x) - g(x)| =
O(((log n)/(n h_n))^{1/2}) almost surely (page 241) but gives NO
explicit constant or critical value for a uniform band.  No uniform
band is therefore returned; the uniform rate is reported as a
diagnostic only.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["npconfband", "horowitz_confidence_bands"]

_BGAUSS = 1.0 / (2.0 * np.sqrt(np.pi))


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def npconfband(x, y, grid=None, h=None, alpha=0.05, s=2, c=1.0, ngrid=25):
    """Nadaraya-Watson fit with pointwise asymptotic confidence bands.

    Parameters
    ----------
    x : array-like, (n,)
    y : array-like, (n,)
    grid : array-like, optional
        Evaluation points; default `ngrid` equally spaced points
        spanning the observed x.
    h : float, optional
        Bandwidth; default c * n**(-1/(2s+1)).
    alpha : float, default 0.05
        Two-sided level; the band is 1 - alpha pointwise.
    s : int, default 2
    c : float, default 1.0
    ngrid : int, default 25

    Returns
    -------
    RichResult
        payload keys: grid, ghat, se, lower, upper, density,
        bandwidth, zcrit, unifrate, alpha, n, method.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = xv.size
    if yv.size != n:
        raise ValueError("x and y must have the same length.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    hh = float(c) * float(n ** (-1.0 / (2 * int(s) + 1))) if h is None else float(h)
    if hh <= 0:
        raise ValueError("bandwidth must be positive.")
    g = (np.linspace(float(np.min(xv)), float(np.max(xv)), int(ngrid))
         if grid is None else np.atleast_1d(np.asarray(grid, dtype=float)))

    K = _gauss((g[:, None] - xv[None, :]) / hh)
    ksum = np.sum(K, axis=1)
    safeks = np.where(ksum > 1e-300, ksum, 1e-300)
    dens = ksum / (n * hh)
    ghat = (K @ yv) / safeks
    resid2 = (yv[None, :] - ghat[:, None]) ** 2
    sig2 = np.sum(K * resid2, axis=1) / safeks
    se = np.sqrt(_BGAUSS * sig2 / (n * hh * np.where(dens > 1e-300, dens, 1e-300)))
    z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    return RichResult(
        title="Nonparametric regression with pointwise confidence bands",
        payload={"grid": g, "ghat": ghat, "se": se,
                 "lower": ghat - z * se, "upper": ghat + z * se,
                 "density": dens, "bandwidth": hh, "zcrit": z,
                 "unifrate": float(np.sqrt(np.log(n) / (n * hh))),
                 "alpha": float(alpha), "n": n,
                 "method": "Horowitz (2009) Appendix A.2.1 pointwise bands (bias uncorrected)"},
    )


horowitz_confidence_bands = npconfband


def cheatsheet():
    return "hrzconf: pointwise confidence bands for a kernel mean regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 400
    xv = np.linspace(0.0, 1.0, n)
    yv = np.sin(3.0 * xv)                    # noiseless
    r = npconfband(xv, yv, h=0.05)
    err = float(np.max(np.abs(r["ghat"][2:-2] - np.sin(3.0 * r["grid"][2:-2]))))
    assert err < 0.05, err
    assert bool(np.all(r["lower"] <= r["ghat"])) and bool(
        np.all(r["upper"] >= r["ghat"]))
    assert abs(r["zcrit"] - 1.959963984540054) < 1e-9, r["zcrit"]
    print("ok", err)
