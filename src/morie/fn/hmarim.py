# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ARIMA(p,d,q) model: ARMA applied to d-th differenced series."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_arima"]


def geron_arima(y, p=1, d=0, q=0, include_mean=True):
    """
    ARIMA(p,d,q) model: ARMA applied to the d-th differenced series.

    Formula: differenced y of order d fit by ARMA(p,q)

    Estimation is Hannan-Rissanen: a long autoregression supplies the
    unobserved innovations, then the ARMA coefficients come from an OLS
    regression of the differenced series on its own lags and those fitted
    innovations. Pure AR models (q=0) skip stage one and are plain OLS.

    Parameters
    ----------
    y : array-like
        Univariate series.
    p, q : int
        AR and MA orders (non-negative).
    d : int
        Differencing order (non-negative, and < len(y)).
    include_mean : bool
        Fit an intercept on the differenced series.

    Returns
    -------
    result : RichResult
        Keys: ar, ma, intercept, residuals, sigma2, aic, fitted, forecast,
        estimate, n, method.

    Examples
    --------
    An exact geometric series is fit exactly by AR(1) with coefficient 0.5:

    >>> y = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    >>> r = geron_arima(y, p=1, d=0, q=0)
    >>> round(float(r["ar"][0]), 9)
    0.5
    >>> round(float(r["sigma2"]), 15)
    0.0

    A perfect linear trend is a constant after one difference, so the
    one-step forecast continues the line:

    >>> r2 = geron_arima([1.0, 3.0, 5.0, 7.0], p=0, d=1, q=0)
    >>> round(float(r2["intercept"]), 9)
    2.0
    >>> [round(float(v), 9) for v in r2["forecast"](2)]
    [9.0, 11.0]

    References
    ----------
    Géron Ch 13
    """
    ys = np.asarray(y, dtype=float).ravel()
    if ys.size == 0:
        raise ValueError("geron_arima: y is empty")
    if not np.all(np.isfinite(ys)):
        raise ValueError("geron_arima: y contains non-finite values")
    P, D, Q = int(p), int(d), int(q)
    if P < 0 or D < 0 or Q < 0:
        raise ValueError("geron_arima: p, d and q must all be non-negative")
    if D >= ys.size:
        raise ValueError(f"geron_arima: cannot difference a length-{ys.size} series {D} times")

    anchors = []
    z = ys
    for _ in range(D):
        anchors.append(float(z[-1]))
        z = np.diff(z)
    m = z.size
    if m <= P + Q:
        raise ValueError(
            f"geron_arima: differenced series has {m} points, too few for ARMA({P},{Q}) "
            f"(need more than {P + Q})"
        )

    def ols(A, b):
        if A.shape[1] == 0:
            return np.zeros(0)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef

    if Q == 0:
        e_hat = None
        start = P
    else:
        k = int(min(max(P + Q + 1, int(np.ceil(np.log(max(m, 2)) ** 2))), max(1, (m - 1) // 2)))
        rows = m - k
        if rows <= k:
            raise ValueError(
                f"geron_arima: series of length {m} is too short for the Hannan-Rissanen "
                f"long autoregression of order {k}"
            )
        Xa = np.column_stack([z[k - i - 1 : m - i - 1] for i in range(k)])
        if include_mean:
            Xa = np.column_stack([np.ones(rows), Xa])
        beta = ols(Xa, z[k:])
        e_hat = np.zeros(m)
        e_hat[k:] = z[k:] - Xa @ beta
        start = max(P, Q, k)

    cols = []
    for i in range(1, P + 1):
        cols.append(z[start - i : m - i])
    if Q > 0:
        for j in range(1, Q + 1):
            cols.append(e_hat[start - j : m - j])
    Xd = np.column_stack(cols) if cols else np.empty((m - start, 0))
    if include_mean:
        Xd = np.column_stack([np.ones(m - start), Xd])
    target = z[start:]
    if Xd.shape[0] <= Xd.shape[1]:
        raise ValueError(
            f"geron_arima: {Xd.shape[0]} usable observations for {Xd.shape[1]} parameters; "
            "shorten the model or lengthen the series"
        )
    coef = ols(Xd, target)

    off = 1 if include_mean else 0
    intercept = float(coef[0]) if include_mean else 0.0
    ar = np.asarray(coef[off : off + P], dtype=float)
    ma = np.asarray(coef[off + P : off + P + Q], dtype=float)

    fitted = Xd @ coef
    resid = target - fitted
    dof = max(len(target) - Xd.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    nobs = len(target)
    kpar = Xd.shape[1] + 1
    aic = float(nobs * np.log(sigma2) + 2 * kpar) if sigma2 > 0 else -np.inf

    def forecast(h=1, _z=z, _ar=ar, _ma=ma, _c=intercept, _res=resid, _anchors=anchors):
        H = int(h)
        if H < 1:
            raise ValueError("forecast: h must be >= 1")
        hist = list(_z)
        errs = list(_res)
        out = []
        for _ in range(H):
            val = _c
            for i, a in enumerate(_ar):
                val += a * hist[-(i + 1)]
            for j, b in enumerate(_ma):
                val += b * (errs[-(j + 1)] if len(errs) > j else 0.0)
            hist.append(val)
            errs.append(0.0)
            out.append(val)
        f = np.asarray(out, dtype=float)
        for anchor in reversed(_anchors):
            f = anchor + np.cumsum(f)
        return f

    return RichResult(
        title=f"ARIMA({P},{D},{Q})",
        summary_lines=[("Intercept", intercept), ("sigma^2", sigma2), ("AIC", aic), ("Observations used", nobs)],
        payload={
            "ar": ar,
            "ma": ma,
            "intercept": intercept,
            "residuals": resid,
            "sigma2": sigma2,
            "aic": aic,
            "fitted": fitted,
            "differenced": z,
            "forecast": forecast,
            "order": (P, D, Q),
            "estimate": sigma2,
            "n": int(nobs),
            "method": f"ARIMA({P},{D},{Q}) by Hannan-Rissanen" if Q else f"ARIMA({P},{D},0) by OLS",
        },
    )


def cheatsheet():
    return "hmarim: ARIMA(p,d,q) model: ARMA applied to d-th differenced series"


# compact alias per ledger/NAMING.md
geronarima = geron_arima
