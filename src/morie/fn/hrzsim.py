# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric weighted nonlinear least squares for a single-index model.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.5.1, equations (2.22)-(2.25) and Theorem 2.2,
equation (2.26) (pages 20-21).  The estimator is Ichimura's (1993).

If G were known, beta would solve the weighted NLS problem

    min_b  S_n(b) = (1/n) sum_i W(X_i) [Y_i - G(X_i'b)]^2      (2.22)

G is not known, so Ichimura replaces it by the kernel estimator

    G_ni(z, b) = (1 / (n h p_ni(z,b))) sum_{j != i}
                    J_nj W(X_j) Y_j K((z - X_j'b) / h)         (2.23)
    p_ni(z, b) = (1 / (n h))          sum_{j != i}
                    J_nj W(X_j)       K((z - X_j'b) / h)       (2.24)

-- leave-one-out, trimmed and weighted the same way as (2.22) -- and
beta solves

    min_btilde  S_n(btilde) = (1/n) sum_i J_i W(X_i)
                              [Y_i - G_ni(X_i'b, b)]^2         (2.25)

over btilde only, which imposes the scale normalisation beta_1 = 1.
Theorem 2.2 then gives n^{1/2}(btilde_n - betatilde) -> N(0, Sigma).

The minimisation uses the shelf's fixed-schedule coordinate search:
a set number of sweeps, a fixed step ladder, no tolerance-based early
exit and no random restart, so the answer is reproducible to the last
bit across languages.
"""

from __future__ import annotations

from . import _array_core as np
from ._horowitz import coord_min

from ._richresult import RichResult

__all__ = ["sindex", "horowitz_single_index_model"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def sindex(x, y, h=None, weights=None, trim=0.01, niter=12, delta=1.0,
           b0=None):
    """Ichimura's semiparametric WNLS estimator of beta and G.

    Parameters
    ----------
    x : array-like, (n, d)
        Covariates, no constant column (location normalisation).
    y : array-like, (n,)
    h : float, optional
        Bandwidth for (2.23)-(2.24).  Default n**(-1/5).
    weights : array-like, (n,), optional
        W(X_i) in (2.22).  Default all ones.
    trim : float, default 0.01
        The constant eta of the trimming set A_x: observation i enters
        (2.25) only when its estimated index density exceeds
        trim * mean(density).  Fixed, not data-tuned.
    niter, delta : int, float
        Fixed schedule of the coordinate search.
    b0 : array-like, (d-1,), optional
        Starting value for betatilde; default the scale-normalised OLS
        slope.

    Returns
    -------
    RichResult
        payload keys: estimate, se, objective, ghat, index, bandwidth,
        ntrim, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("a single-index model needs at least two covariates.")
    hh = float(n ** -0.2) if h is None else float(h)
    W = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()

    def crit(bt, want=False):
        b = np.concatenate([np.array([1.0]), np.asarray(bt, dtype=float)])
        z = X @ b
        u = (z[:, None] - z[None, :]) / hh
        K = _gauss(u) * W[None, :]
        np.fill_diagonal(K, 0.0)                       # leave one out
        den = np.sum(K, axis=1) / (n * hh)             # (2.24)
        num = (K @ yv) / (n * hh)                      # (2.23)
        safe = np.where(den > 1e-300, den, 1e-300)
        gh = num / safe
        keep = den > float(trim) * float(np.mean(den))  # J_i
        r = yv - gh
        val = float(np.sum(np.where(keep, W * r * r, 0.0))) / n
        if want:
            return val, gh, z, keep, r
        return val

    if b0 is None:
        ols = np.linalg.lstsq(X, yv, rcond=None)[0]
        start = (ols[1:] / ols[0]) if abs(float(ols[0])) > 1e-12 else np.zeros(d - 1)
    else:
        start = np.asarray(b0, dtype=float).ravel()

    bt, obj = coord_min(crit, list(start), niter=int(niter), delta=float(delta))
    bt = np.asarray(bt, dtype=float)
    val, gh, z, keep, r = crit(bt, want=True)
    beta = np.concatenate([np.array([1.0]), bt])

    # Sigma from the outer product of the numerical gradient of the
    # per-observation residual with respect to btilde (Theorem 2.2).
    eps = 1e-5
    Jm = np.zeros((n, d - 1))
    for j in range(d - 1):
        bp = bt.copy()
        bp[j] = bp[j] + eps
        _, gp, _, _, _ = crit(bp, want=True)
        bm = bt.copy()
        bm[j] = bm[j] - eps
        _, gmn, _, _, _ = crit(bm, want=True)
        Jm[:, j] = -(gp - gmn) / (2.0 * eps)
    kf = keep.astype(float)
    A = (Jm * (kf * W)[:, None]).T @ Jm / n
    s2 = float(np.sum(kf * W * r * r) / max(float(np.sum(kf)), 1.0))
    try:
        cov = s2 * np.linalg.inv(A + 1e-12 * np.eye(d - 1)) / n
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        se = np.full(d - 1, np.nan)
    return RichResult(
        title="Semiparametric WNLS single-index model (eq. 2.25)",
        payload={"estimate": beta, "se": np.concatenate([np.array([0.0]), se]),
                 "objective": float(val), "ghat": gh, "index": z,
                 "bandwidth": hh, "ntrim": int(n - np.sum(kf)), "n": n,
                 "method": "Horowitz (2009) eq. (2.25), Ichimura semiparametric WNLS"},
    )


horowitz_single_index_model = sindex


def cheatsheet():
    return "hrzsim: Ichimura semiparametric WNLS single-index estimator (eq. 2.25)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 150
    X = np.column_stack([np.linspace(-2, 2, n), np.cos(np.arange(1, n + 1) * 0.9)])
    z = X @ np.array([1.0, 0.8])
    y = z / (1.0 + np.abs(z))                    # smooth monotone G
    r = sindex(X, y, h=0.35)
    got = float(r["estimate"][1])
    assert abs(got - 0.8) < 0.2, got
    assert r["objective"] < 1e-3, r["objective"]
    print("ok", got, r["objective"])
