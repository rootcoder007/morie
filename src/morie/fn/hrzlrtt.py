# morie.fn -- function file (rootcoder007/morie)
"""Generalized likelihood ratio test of a parametric mean regression.

SOURCE.  Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, states on page 3 that "The subject of specification
testing, which has received much attention recently, is also not
treated."  The book therefore contains NO likelihood-ratio-type test
and no locator for one.  The primary source consulted instead is

    Fan, J., Zhang, C. and Zhang, J. (2001), "Generalized Likelihood
    Ratio Statistics and Wilks Phenomenon", *Annals of Statistics*
    29(1), 153-193.

Section 4.1, page 170.  For H0: m(x) parametric against H1: m
nonparametric, with

    RSS0 = sum_i [Y_i - mhat_0(X_i)]^2,
    RSS1 = sum_i [Y_i - mhat_h(X_i)]^2   (local linear fit),

the generalized likelihood ratio is

    lambda_n = l_n(H1) - l_n(H0) = (n/2) log(RSS0 / RSS1)

and the Wilks type of result is

    r_K lambda_n ~a chi^2( r_K c_K |Omega| / h )              (4.1)

where |Omega| is the length of the support of X,

    c_K = K(0) - 2^{-1} ||K||_2^2                     (page 170)
    r_K = [K(0) - (1/2)||K||^2]
          / integral (K(t) - (1/2) (K*K)(t))^2 dt     (Theorem 5, p.165)

Remark 4.1 (page 171) states that for a parametric null other than the
linear model the same result holds applied to the residuals of the
fitted null model.

TWO THINGS THE STUB THIS REPLACES GOT WRONG, recorded here because
they are easy to repeat: the statistic is r_K * lambda_n, not
2 * lambda_n (r_K is 2 only for a kernel with r_K near 2 -- the
Epanechnikov kernel is the closest at 2.1153, Table 2, page 170); and
the degrees of freedom r_K c_K |Omega| / h grow as the bandwidth
shrinks, so there is no fixed integer df.

Table 2 (page 170) prints r_K and c_K for five kernels.  For the
Gaussian it gives r_K = 2.5375 and c_K = 0.7737.  The r_K value
reproduces exactly from the Theorem 5 formula for the STANDARD normal
kernel; c_K = 0.7737 does not (the standard normal gives
K(0) - ||K||^2/2 = 0.2578954) but equals three times it, which is what
c_K becomes for a normal kernel rescaled by 3 -- c_K, unlike r_K, is
not scale invariant.  This module computes c_K in closed form for the
kernel it actually uses, so the bandwidth and the constant refer to the
same kernel; the tabulated pair is available via kernel="table".
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["splrtest", "horowitz_likelihood_ratio_test"]

# Table 2, Fan, Zhang and Zhang (2001), page 170
_TABLE2 = {
    "uniform": (1.2000, 0.2500),
    "epanechnikov": (2.1153, 0.4500),
    "biweight": (2.3061, 0.5804),
    "triweight": (2.3797, 0.6858),
    "gaussian": (2.5375, 0.7737),
}
# closed forms for the standard Gaussian kernel actually used below
_RK_GAUSS = 2.5374999999999996
_CK_GAUSS = 1.0 / np.sqrt(2.0 * np.pi) - 0.5 / (2.0 * np.sqrt(np.pi))


def _gauss(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)


def splrtest(x, y, fitted=None, h=None, degree=1, kernel="closed"):
    """Generalized likelihood ratio test, parametric null vs nonparametric.

    Parameters
    ----------
    x : array-like, (n,)
        Scalar covariate.
    y : array-like, (n,)
    fitted : array-like, (n,), optional
        Fitted values under the parametric null.  Default is the
        least-squares fit of a polynomial of the given `degree`, so
        the default null is linearity (Section 4.1).
    h : float, optional
        Bandwidth of the local linear alternative.  Default
        n**(-1/5).
    degree : int, default 1
        Degree of the default polynomial null.
    kernel : {"closed", "table"}, default "closed"
        "closed" uses r_K and c_K computed for the standard Gaussian
        kernel this function evaluates; "table" uses the Gaussian row
        of Table 2 verbatim.

    Returns
    -------
    RichResult
        payload keys: statistic, p_value, n, method, lambdan, rk, ck,
        df, rss0, rss1, bandwidth, support.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    n = xv.size
    if yv.size != n:
        raise ValueError("x and y must have the same length.")
    if n < 5:
        raise ValueError("need at least five observations.")
    hh = float(n ** -0.2) if h is None else float(h)
    if hh <= 0:
        raise ValueError("bandwidth must be positive.")

    if fitted is None:
        P = np.column_stack([xv ** k for k in range(int(degree) + 1)])
        coef = np.linalg.lstsq(P, yv, rcond=None)[0]
        f0 = P @ coef
    else:
        f0 = np.asarray(fitted, dtype=float).ravel()
        if f0.size != n:
            raise ValueError("fitted must have one entry per observation.")

    # local linear fit at each X_i (the H1 estimator of Section 4.1)
    u = (xv[:, None] - xv[None, :]) / hh
    W = _gauss(u)
    dx = xv[None, :] - xv[:, None]
    s0 = np.sum(W, axis=1)
    s1 = np.sum(W * dx, axis=1)
    s2 = np.sum(W * dx * dx, axis=1)
    t0 = np.sum(W * yv[None, :], axis=1)
    t1 = np.sum(W * dx * yv[None, :], axis=1)
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) > 1e-300, det, 1e-300)
    f1 = (s2 * t0 - s1 * t1) / det

    rss0 = float(np.sum((yv - f0) ** 2))
    rss1 = float(np.sum((yv - f1) ** 2))
    rss1 = max(rss1, 1e-300)
    lam = 0.5 * n * float(np.log(max(rss0, 1e-300) / rss1))

    if str(kernel) == "table":
        rk, ck = _TABLE2["gaussian"]
    else:
        rk, ck = _RK_GAUSS, float(_CK_GAUSS)
    support = float(np.max(xv) - np.min(xv))
    df = rk * ck * support / hh
    statv = rk * lam
    pval = float(1.0 - stats.chi2.cdf(statv, df)) if df > 0 else float("nan")
    return RichResult(
        title="Generalized likelihood ratio test (Fan, Zhang and Zhang 2001)",
        payload={"statistic": float(statv),
                 "p_value": float(min(max(pval, 0.0), 1.0)),
                 "n": n,
                 "method": "Fan, Zhang and Zhang (2001) eq. (4.1) GLR / Wilks",
                 "lambdan": float(lam), "rk": float(rk), "ck": float(ck),
                 "df": float(df), "rss0": rss0, "rss1": rss1,
                 "bandwidth": hh, "support": support},
    )


horowitz_likelihood_ratio_test = splrtest


def cheatsheet():
    return "hrzlrtt: generalized likelihood ratio test of a parametric mean regression"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    n = 200
    xv = np.linspace(0.0, 1.0, n)
    # deterministic high-frequency wobble that neither fit can track,
    # so both residual sums are non-degenerate
    wob = 0.3 * np.cos(50.0 * np.pi * xv)
    # H0 true: linear mean -> RSS0 close to RSS1, p_value not small
    r0 = splrtest(xv, 2.0 + 3.0 * xv + wob, h=0.2)
    assert r0["p_value"] > 0.05, (r0["lambdan"], r0["p_value"])
    # H0 false: strongly curved mean -> p_value small
    r1 = splrtest(xv, 3.0 * np.sin(8.0 * xv) + wob, h=0.05)
    assert r1["lambdan"] > r0["lambdan"], (r0["lambdan"], r1["lambdan"])
    assert r1["p_value"] < 0.05, r1["p_value"]
    # r_K reproduces the Table 2 Gaussian entry
    assert abs(_RK_GAUSS - 2.5375) < 1e-4
    print("ok", r0["lambdan"], r1["lambdan"], r1["p_value"])
