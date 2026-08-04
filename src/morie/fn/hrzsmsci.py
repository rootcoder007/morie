# morie.fn -- function file (rootcoder007/morie)
"""Confidence intervals for the smoothed maximum-score estimator.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 4.3.3: equation (4.25) (page 109), Theorem 4.5
(page 113), Theorem 4.6 and equation (4.32) (pages 114-115).

The smoothed maximum-score estimator solves

    maximize_{|b1|=1, b in B}  S_sms(b) = (1/n) sum_i (2Y_i - 1)
                                          K(X_i'b / h)          (4.25)

with K a twice-differentiable function rising from 0 to 1 (the
integral of a kernel).  Theorem 4.5(b) gives

    (n h)^{1/2} (btilde_n - betatilde)
        -> N(-lambda^{1/2} Q^{-1} A, Q^{-1} D Q^{-1})

and Theorem 4.6 supplies consistent estimators

    D_n = (1/(n h)) sum_i Xt_i Xt_i' K'(X_i'b_n / h)^2
    Q_n = d^2 S_sms(b_n) / d btilde d btilde'
    A_n = (1/(n h*^{s+1})) sum_i (2Y_i - 1) Xt_i K'(X_i'b_n / h*)

so that V_n = Q_n^{-1} D_n Q_n^{-1} estimates the asymptotic
covariance (4.32), the bias-corrected estimator is

    bhat_n = btilde_n + (lambda/n)^{s/(2s+1)} Q_n^{-1} A_n

and t = (n h)^{1/2} (bhat_nj - betatilde_j) / V_nj^{1/2} is
asymptotically N(0,1) (page 115).  Intervals follow from that.

NOTE on the alternative route.  Subsampling (Politis and Romano 1994;
Politis et al. 1999; Delgado et al. 2001) is the book's remedy on
page 107 for the UNSMOOTHED maximum-score estimator, whose n^{-1/3}
limit is nonstandard and for which the bootstrap fails (Abrevaya and
Huang 2005).  The smoothed estimator does not need it: its limit is
normal and the interval above is analytic.  An analytic interval is
also the only one compatible with this shelf's determinism rule, since
subsampling would draw random subsets.

The maximisation uses the shelf's fixed-schedule coordinate search:
fixed sweeps, fixed step ladder, no tolerance-based early exit, no
random restarts.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._horowitz import coord_min

from ._richresult import RichResult

__all__ = ["smsciband", "horowitz_sms_confidence"]


def _Kcdf(v):
    return stats.norm.cdf(v)


def _Kprime(v):
    return np.exp(-0.5 * v * v) / np.sqrt(2.0 * np.pi)


def smsciband(x, y, h=None, alpha=0.05, s=2, hstar=None, biascorrect=True,
              niter=12, delta=1.0, b0=None):
    """Smoothed maximum-score estimate with analytic confidence intervals.

    Parameters
    ----------
    x : array-like, (n, d)
    y : array-like, (n,) of 0/1
    h : float, optional
        Bandwidth in (4.25).  Default n**(-1/(2s+1)), which is the
        rate hn proportional to n^{-1/(2s+1)} of Theorem 4.5(c).
    alpha : float, default 0.05
    s : int, default 2
        Smoothness order; the rate is n^{-s/(2s+1)}.
    hstar : float, optional
        The separate bandwidth h* of Theorem 4.6 used for A_n.
        Default n**(-0.5/(2s+1)), i.e. delta = 0.5.
    biascorrect : bool, default True
        Use bhat_n rather than btilde_n at the centre of the interval.
    niter, delta : int, float
        Fixed coordinate-search schedule.
    b0 : array-like, (d-1,), optional

    Returns
    -------
    RichResult
        payload keys: estimate, biascorrected, se, lower, upper, tstat,
        vcov, bandwidth, hstar, zcrit, objective, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    n, d = X.shape
    if d < 2:
        raise ValueError("need at least two covariates for a scale normalisation.")
    uy = np.unique(yv)
    if bool(np.any((uy != 0.0) & (uy != 1.0))):
        raise ValueError("y must be binary 0/1 for a binary-response model.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    si = int(s)
    hh = float(n ** (-1.0 / (2 * si + 1))) if h is None else float(h)
    hs = float(n ** (-0.5 / (2 * si + 1))) if hstar is None else float(hstar)
    sgn = 2.0 * yv - 1.0
    Xt = X[:, 1:]

    def negS(bt):
        b = np.concatenate([np.array([1.0]), np.asarray(bt, dtype=float)])
        return -float(np.sum(sgn * _Kcdf((X @ b) / hh))) / n

    if b0 is None:
        ols = np.linalg.lstsq(X, yv, rcond=None)[0]
        start = (ols[1:] / ols[0]) if abs(float(ols[0])) > 1e-12 else np.zeros(d - 1)
    else:
        start = np.asarray(b0, dtype=float).ravel()
    bt, obj = coord_min(negS, list(start), niter=int(niter), delta=float(delta))
    bt = np.asarray(bt, dtype=float)
    beta = np.concatenate([np.array([1.0]), bt])
    z = X @ beta

    # D_n (Theorem 4.6)
    kp = _Kprime(z / hh)
    Dn = (Xt * (kp * kp)[:, None]).T @ Xt / (n * hh)

    # Q_n: second derivative of S_sms, by central differences on the
    # analytic first derivative dS/dbtilde_j = (1/(n h)) sum sgn Xt_j K'
    eps = 1e-5

    def grad(btv):
        b = np.concatenate([np.array([1.0]), np.asarray(btv, dtype=float)])
        kpv = _Kprime((X @ b) / hh)
        return (Xt * (sgn * kpv)[:, None]).sum(axis=0) / (n * hh)

    Qn = np.zeros((d - 1, d - 1))
    for j in range(d - 1):
        bp = bt.copy()
        bp[j] = bp[j] + eps
        bm = bt.copy()
        bm[j] = bm[j] - eps
        Qn[:, j] = (grad(bp) - grad(bm)) / (2.0 * eps)
    Qn = 0.5 * (Qn + Qn.T)

    # A_n (Theorem 4.6), on its own bandwidth h*
    An = (Xt * (sgn * _Kprime(z / hs))[:, None]).sum(axis=0) / (n * hs ** (si + 1))

    try:
        Qi = np.linalg.inv(Qn + 1e-12 * np.eye(d - 1))
    except Exception:
        Qi = np.linalg.pinv(Qn)
    Vn = Qi @ Dn @ Qi                                            # (4.32)
    se = np.sqrt(np.maximum(np.diag(Vn), 0.0) / (n * hh))
    bhat = bt + (1.0 / n) ** (si / float(2 * si + 1)) * (Qi @ An)
    centre = bhat if biascorrect else bt
    zc = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
    return RichResult(
        title="Smoothed maximum score with analytic confidence intervals",
        payload={"estimate": beta,
                 "biascorrected": np.concatenate([np.array([1.0]), bhat]),
                 "se": se, "lower": centre - zc * se,
                 "upper": centre + zc * se,
                 "tstat": np.where(se > 0, centre / np.where(se > 0, se, 1.0),
                                   np.nan),
                 "vcov": Vn, "bandwidth": hh, "hstar": hs, "zcrit": zc,
                 "objective": -float(obj), "n": n,
                 "method": "Horowitz (2009) Theorem 4.6, eq. (4.32) analytic CI"},
    )


horowitz_sms_confidence = smsciband


def cheatsheet():
    return "hrzsmsci: analytic confidence intervals for smoothed maximum score"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    X = np.column_stack([np.linspace(-2, 2, n),
                         np.cos(np.arange(1, n + 1) * 0.8)])
    yv = ((X @ np.array([1.0, 0.6])) >= 0.0).astype(float)
    r = smsciband(X, yv, h=0.3)
    assert abs(float(r["estimate"][0]) - 1.0) < 1e-12
    assert bool(np.all(r["se"] >= 0.0))
    assert bool(np.all(r["lower"] <= r["upper"]))
    print("ok", r["estimate"], r["se"])
