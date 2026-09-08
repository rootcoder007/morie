# morie.fn -- function file (rootcoder007/morie)
"""Rate and asymptotic distribution of the index-function estimator G.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.4, equations (2.15)-(2.18) (pages 17-18),
with the limit constants from Appendix A.2 (page 241).

The kernel estimator of G at a point z of the index support is

    G_n(z) = (1 / (n h_n p_n(z))) sum_i Y_i K((z - X_i'b_n) / h_n)  (2.17)
    p_n(z) = (1 / (n h_n))        sum_i     K((z - X_i'b_n) / h_n)  (2.18)

Because b_n converges at n^{-1/2}, faster than any nonparametric rate,

    (n h_n)^{1/2} [G_n(z) - G(z)]
        = (n h_n)^{1/2} [G_n^*(z) - G(z)] + o_p(1)

where G_n^* uses the true beta.  So estimating beta costs nothing
asymptotically, and G converges at the one-dimensional nonparametric
rate n^{-s/(2s+1)} with h_n = c n^{-1/(2s+1)} and

    n^{s/(2s+1)} [G_n(z) - G(z)] -> N(mu_R, sigma_R^2),
    sigma_R^2 = B sigma^2(z) / (c p(z)),  B = integral K(v)^2 dv.

This function evaluates (2.17)-(2.18) at a supplied grid and returns
the pointwise standard error implied by sigma_R^2 above.  The
bandwidth is either given explicitly or set by the fixed formula
h = c n^{-1/(2s+1)}: nothing is cross-validated and nothing is random.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["simgrate", "horowitz_rate_G_estimation"]

# B = integral K(v)^2 dv for the standard Gaussian kernel
_BGAUSS = 1.0 / (2.0 * np.sqrt(np.pi))


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def simgrate(x, y, beta, grid=None, h=None, s=2, c=1.0, ngrid=25):
    """Kernel estimate of G on the index, with rate and pointwise SE.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,)
    beta : array-like, (d,)
        Index coefficients (scale normalised, beta[0] = 1).
    grid : array-like, optional
        Points of the index at which to evaluate G.  Default is
        `ngrid` equally spaced points spanning the observed index.
    h : float, optional
        Bandwidth.  Default c * n**(-1/(2s+1)).
    s : int, default 2
    c : float, default 1.0
    ngrid : int, default 25

    Returns
    -------
    RichResult
        payload keys: grid, ghat, se, density, bandwidth, exponent,
        rate, effn, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    if X.shape[1] != b.size and X.shape[0] == b.size:
        X = X.T
    n = X.shape[0]
    if yv.size != n:
        raise ValueError("y must have one entry per row of x.")
    hh = float(c) * float(n ** (-1.0 / (2 * int(s) + 1))) if h is None else float(h)
    if hh <= 0:
        raise ValueError("bandwidth must be positive.")

    z = X @ b
    g = (np.linspace(float(np.min(z)), float(np.max(z)), int(ngrid))
         if grid is None else np.atleast_1d(np.asarray(grid, dtype=float)))
    K = _gauss((g[:, None] - z[None, :]) / hh)
    dens = np.sum(K, axis=1) / (n * hh)                       # (2.18)
    num = (K @ yv) / (n * hh)
    safe = np.where(dens > 1e-300, dens, 1e-300)
    ghat = num / safe                                         # (2.17)

    # local residual variance sigma^2(z), kernel weighted
    resid2 = (yv[None, :] - ghat[:, None]) ** 2
    sig2 = np.sum(K * resid2, axis=1) / np.where(
        np.sum(K, axis=1) > 1e-300, np.sum(K, axis=1), 1e-300)
    se = np.sqrt(_BGAUSS * sig2 / (n * hh * safe))
    expo = int(s) / float(2 * int(s) + 1)
    return RichResult(
        title="Kernel estimate of G on the index (eq. 2.17)",
        payload={"grid": g, "ghat": ghat, "se": se, "density": dens,
                 "bandwidth": hh, "exponent": expo,
                 "rate": float(n ** (-expo)), "effn": float(n * hh),
                 "n": n,
                 "method": "Horowitz (2009) eq. (2.17)-(2.18), rate n^{-s/(2s+1)}"},
    )


horowitz_rate_G_estimation = simgrate


def cheatsheet():
    return "hrzrateG: kernel estimate of the index function G and its (nh)^{1/2} rate"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 400
    x1 = np.linspace(-2, 2, n)
    x2 = np.cos(np.arange(n) * 0.7)
    X = np.column_stack([x1, x2])
    beta = np.array([1.0, 0.5])
    z = X @ beta
    y = z ** 2                      # G(v) = v^2, noiseless
    r = simgrate(X, y, beta, h=0.15)
    err = float(np.max(np.abs(r["ghat"][3:-3] - r["grid"][3:-3] ** 2)))
    assert err < 0.15, err
    assert abs(r["exponent"] - 0.4) < 1e-12
    print("ok", err)
