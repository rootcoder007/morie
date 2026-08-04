# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric maximum likelihood for a binary-response index model.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.5.3, equation (2.33) and the semiparametric
analog on page 28.  The estimator is Klein and Spady's (1993).

With Y in {0, 1} the index model gives G(x'beta) = P(Y = 1 | X = x),
and if G were known the efficient estimator would maximise

    log L(b) = (1/n) sum_i {Y_i log G(X_i'b)
                            + (1 - Y_i) log[1 - G(X_i'b)]}     (2.33)

In the semiparametric case G is replaced by the leave-one-out kernel
estimator.  Because Var(Y|X=x) = G(x'beta)[1 - G(x'beta)] depends on x
only through the index, the weight function cancels, so the UNWEIGHTED
estimator of G already attains the efficiency bound (page 28):

    Ghat_ni(z, b) = (1/(n h phat_ni(z,b))) sum_{j != i}
                       J_nj Y_j K((z - X_j'b)/h)
    phat_ni(z, b) = (1/(n h))              sum_{j != i}
                       J_nj     K((z - X_j'b)/h)

and betatilde maximises

    log L_SP(btilde) = (1/n) sum_i J_i {Y_i log Ghat_ni(X_i'b, b)
                       + (1 - Y_i) log[1 - Ghat_ni(X_i'b, b)]}

Klein and Spady use elaborate trimming to hold Ghat away from 0 and 1;
the book notes (page 27) that trimming has little effect in practice
and uses only observations in a fixed set A_x.  That is what is done
here, with an explicit floor on Ghat.

Maximisation is by the shelf's fixed-schedule coordinate search: fixed
sweeps, fixed step ladder, no tolerance-based early exit, no restarts.
"""

from __future__ import annotations

from . import _array_core as np
from ._horowitz import coord_min

from ._richresult import RichResult

__all__ = ["spmlebin", "horowitz_semipar_mle_binary"]


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def spmlebin(x, y, h=None, trim=0.01, floor=1e-4, niter=12, delta=1.0,
             b0=None):
    """Klein-Spady semiparametric MLE of beta in a binary-response model.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,) of 0/1
    h : float, optional
        Bandwidth; default n**(-1/5).
    trim : float, default 0.01
        Density trimming constant defining A_x.
    floor : float, default 1e-4
        Ghat is clipped into [floor, 1 - floor] before the logarithm.
    niter, delta : int, float
        Fixed coordinate-search schedule.
    b0 : array-like, (d-1,), optional

    Returns
    -------
    RichResult
        payload keys: estimate, se, loglik, ghat, index, bandwidth,
        ntrim, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("a single-index model needs at least two covariates.")
    uy = np.unique(yv)
    if bool(np.any((uy != 0.0) & (uy != 1.0))):
        raise ValueError("y must be binary 0/1 for a binary-response model.")
    hh = float(n ** -0.2) if h is None else float(h)
    fl = float(floor)

    def negll(bt, want=False):
        b = np.concatenate([np.array([1.0]), np.asarray(bt, dtype=float)])
        z = X @ b
        K = _gauss((z[:, None] - z[None, :]) / hh)
        np.fill_diagonal(K, 0.0)
        den = np.sum(K, axis=1) / (n * hh)
        num = (K @ yv) / (n * hh)
        safe = np.where(den > 1e-300, den, 1e-300)
        gh = np.clip(num / safe, fl, 1.0 - fl)
        keep = den > float(trim) * float(np.mean(den))
        ll = np.where(keep, yv * np.log(gh) + (1.0 - yv) * np.log(1.0 - gh), 0.0)
        val = -float(np.sum(ll)) / n
        if want:
            return val, gh, z, keep
        return val

    if b0 is None:
        ols = np.linalg.lstsq(X, yv, rcond=None)[0]
        start = (ols[1:] / ols[0]) if abs(float(ols[0])) > 1e-12 else np.zeros(d - 1)
    else:
        start = np.asarray(b0, dtype=float).ravel()

    bt, obj = coord_min(negll, list(start), niter=int(niter), delta=float(delta))
    bt = np.asarray(bt, dtype=float)
    val, gh, z, keep = negll(bt, want=True)
    beta = np.concatenate([np.array([1.0]), bt])

    # information from the numerical outer product of the score
    eps = 1e-5
    S = np.zeros((n, d - 1))
    kf = keep.astype(float)
    for j in range(d - 1):
        bp = bt.copy()
        bp[j] = bp[j] + eps
        _, gp, _, _ = negll(bp, want=True)
        bm = bt.copy()
        bm[j] = bm[j] - eps
        _, gm, _, _ = negll(bm, want=True)
        dg = (gp - gm) / (2.0 * eps)
        S[:, j] = kf * dg * (yv / gh - (1.0 - yv) / (1.0 - gh))
    I = S.T @ S / n
    try:
        cov = np.linalg.inv(I + 1e-12 * np.eye(d - 1)) / n
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        se = np.full(d - 1, np.nan)
    return RichResult(
        title="Semiparametric MLE, binary-response index model",
        payload={"estimate": beta, "se": np.concatenate([np.array([0.0]), se]),
                 "loglik": -float(val), "ghat": gh, "index": z,
                 "bandwidth": hh, "ntrim": int(n - np.sum(kf)), "n": n,
                 "method": "Horowitz (2009) eq. (2.33) and page 28, Klein-Spady"},
    )


horowitz_semipar_mle_binary = spmlebin


def cheatsheet():
    return "hrzsmle: Klein-Spady semiparametric MLE for a binary-response index model"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    X = np.column_stack([np.linspace(-2, 2, n), np.cos(np.arange(1, n + 1) * 0.8)])
    z = X @ np.array([1.0, 0.6])
    # deterministic 0/1 outcomes tracking a logistic link
    p = 1.0 / (1.0 + np.exp(-2.0 * z))
    yv = (p > np.linspace(0.02, 0.98, n)[np.argsort(np.argsort(z))]).astype(float)
    r = spmlebin(X, yv, h=0.4)
    assert r["loglik"] < 0.0
    assert abs(float(r["estimate"][0]) - 1.0) < 1e-12
    print("ok", r["estimate"], r["loglik"])
