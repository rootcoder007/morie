# morie.fn -- function file (rootcoder007/morie)
"""Shared cointegration core: ADF, Engle-Granger, Johansen, VECM.

References
----------
Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error
correction: representation, estimation, and testing. *Econometrica*,
55(2), 251-276.

Johansen, S. (1991). Estimation and hypothesis testing of
cointegration vectors in Gaussian vector autoregressive models.
*Econometrica*, 59(6), 1551-1580.

Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University
Press. Ch. 19 (cointegration), Ch. 20 (full-information ML).

MacKinnon, J. G. (2010). Critical values for cointegration tests.
Queen's Economics Department Working Paper 1227.
"""

from . import _array_core as np

__all__ = ["adf_test", "engle_granger", "johansen", "vecm_fit"]

# MacKinnon (2010) Table 2 asymptotic critical values, response-surface
# intercept (tau_infinity) for the constant-only case, by number of
# I(1) variables in the cointegrating regression.
_MACKINNON_TAU = {
    1: {0.01: -3.43035, 0.05: -2.86154, 0.10: -2.56677},
    2: {0.01: -3.89796, 0.05: -3.33613, 0.10: -3.04445},
    3: {0.01: -4.29374, 0.05: -3.74066, 0.10: -3.45218},
    4: {0.01: -4.64405, 0.05: -4.09600, 0.10: -3.81020},
    5: {0.01: -4.95277, 0.05: -4.41310, 0.10: -4.13179},
}


def _lag(x, k):
    return x[:-k] if k else x


def adf_test(x, lags=1, trend="c"):
    r"""Augmented Dickey-Fuller test.

    Dickey, D. A. and Fuller, W. A. (1979), "Distribution of the
    estimators for autoregressive time series with a unit root",
    Journal of the American Statistical Association 74(366),
    427-431 -- the test. Said, S. E. and Dickey, D. A. (1984),
    "Testing for unit roots in autoregressive-moving average
    models of unknown order", Biometrika 71(3), 599-607 -- the
    augmentation that makes it the ADF. PDF not in hand (JSTOR and OUP serve HTML); cited from bibliographic details.
    The critical values used below are MacKinnon (2010), cited
    in full at the top of this module.


    Regresses :math:`\Delta x_t` on :math:`x_{t-1}`, the chosen
    deterministic terms and ``lags`` lagged differences; the statistic
    is the t-ratio on :math:`x_{t-1}`. Under the null of a unit root
    that ratio does NOT have a t distribution, which is why the
    critical values come from MacKinnon's response surface rather than
    a normal table.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    lags = int(lags)
    if lags < 0:
        raise ValueError(f"lags must be non-negative, got {lags}.")
    if n < lags + 10:
        raise ValueError(f"series too short for {lags} lags (n = {n}).")
    if trend not in ("n", "c", "ct"):
        raise ValueError("trend must be 'n', 'c' or 'ct'.")

    dx = np.diff(x)
    T = dx.size - lags
    y = dx[lags:]
    cols = [x[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i : -i] if i else dx[lags:])
    if trend in ("c", "ct"):
        cols.append(np.ones(T))
    if trend == "ct":
        cols.append(np.arange(T, dtype=float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = T - X.shape[1]
    if dof <= 0:
        raise ValueError("not enough observations for this specification.")
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(s2 * xtx_inv[0, 0])
    stat = float(beta[0] / se)
    return {"statistic": stat, "n_obs": int(T), "lags": lags, "trend": trend}


def _tau_pvalue(stat, k):
    """Bracketed lookup against MacKinnon's critical values.

    Returns an interval, not a fabricated exact p-value: interpolating
    a p-value off three critical points would look more precise than
    the table supports.
    """
    cv = _MACKINNON_TAU[min(max(k, 1), 5)]
    if stat < cv[0.01]:
        return "< 0.01", cv
    if stat < cv[0.05]:
        return "0.01 - 0.05", cv
    if stat < cv[0.10]:
        return "0.05 - 0.10", cv
    return "> 0.10", cv


def engle_granger(y, X, lags=1):
    r"""Engle-Granger two-step cointegration test.

    Step 1 regresses :math:`y_t` on :math:`X_t` by OLS; step 2 runs an
    ADF test on the residual. Because the residual is *estimated*, the
    ADF statistic must be compared against MacKinnon critical values
    indexed by the number of variables -- using standard ADF values
    here over-rejects, which is the classic error this function
    guards against.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y.size:
        raise ValueError("y and X must have the same number of observations.")
    if y.size < 20:
        raise ValueError(f"need at least 20 observations, got {y.size}.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X))):
        raise ValueError("y and X must be finite.")

    D = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ beta
    adf = adf_test(resid, lags=lags, trend="n")
    k = X.shape[1] + 1
    band, cv = _tau_pvalue(adf["statistic"], k)
    return {
        "beta": beta[1:], "intercept": float(beta[0]), "residuals": resid,
        "adf_stat": adf["statistic"], "p_value_band": band,
        "critical_values": cv, "n_vars": int(k),
        "cointegrated_5pct": bool(adf["statistic"] < cv[0.05]),
        "lags": int(lags), "n": int(y.size),
        "method": "Engle-Granger two-step (MacKinnon 2010 critical values)",
    }


# Johansen trace-test critical values, Osterwald-Lenum (1992) Table 1
# (constant in the cointegrating relation), by number of common trends
# n - r. Rows are 90%, 95%, 99%.
_TRACE_CV = {
    1: (2.71, 3.84, 6.65), 2: (13.31, 15.34, 19.69), 3: (26.70, 29.38, 35.65),
    4: (43.84, 47.21, 54.46), 5: (64.74, 68.52, 76.07), 6: (89.37, 94.15, 103.18),
}


def johansen(Y, lags=1):
    r"""Johansen trace test by reduced-rank regression.

    Concentrates the VECM by regressing :math:`\Delta Y_t` and
    :math:`Y_{t-1}` on the lagged differences, then solves the
    generalised eigenvalue problem
    :math:`|\lambda S_{11} - S_{10}S_{00}^{-1}S_{01}| = 0`. The trace
    statistic for rank r is
    :math:`-T \sum_{i>r} \ln(1 - \lambda_i)`.

    Unlike Engle-Granger this finds *all* cointegrating vectors and
    does not depend on which variable is placed on the left.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2-D (T observations x n series).")
    T0, n = Y.shape
    if n < 2:
        raise ValueError("need at least 2 series.")
    if n > 6:
        raise ValueError("critical values are tabulated for at most 6 series.")
    lags = int(lags)
    if lags < 1:
        raise ValueError(f"lags must be at least 1, got {lags}.")
    if T0 < 10 * lags + 20:
        raise ValueError(f"series too short for {lags} lags (T = {T0}).")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must be finite.")

    dY = np.diff(Y, axis=0)
    T = dY.shape[0] - lags
    R0y = dY[lags:]
    R1y = Y[lags:-1]
    Z = [np.ones((T, 1))]
    for i in range(1, lags + 1):
        Z.append(dY[lags - i : -i])
    Z = np.column_stack(Z)
    P = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    R0 = R0y - P @ R0y
    R1 = R1y - P @ R1y

    S00 = R0.T @ R0 / T
    S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T
    M = np.linalg.pinv(S11) @ S01.T @ np.linalg.pinv(S00) @ S01
    ev, evec = np.linalg.eig(M)
    ev = np.real(ev)
    evec = np.real(evec)
    order = np.argsort(ev)[::-1]
    ev = np.clip(ev[order], 0, 1 - 1e-12)
    evec = evec[:, order]

    trace = np.array([-T * np.sum(np.log(1 - ev[r:])) for r in range(n)])
    maxeig = np.array([-T * np.log(1 - ev[r]) for r in range(n)])
    cv = np.array([_TRACE_CV[n - r] for r in range(n)])

    rank = 0
    for r in range(n):
        if trace[r] > cv[r][1]:  # 95%
            rank = r + 1
        else:
            break

    # normalise each vector so its largest-loading element is 1
    B = evec.copy()
    for j in range(n):
        idx = np.argmax(np.abs(B[:, j]))
        if B[idx, j] != 0:
            B[:, j] = B[:, j] / B[idx, j]

    return {
        "eigenvalues": ev, "trace_stat": trace, "max_eig_stat": maxeig,
        "trace_crit_90_95_99": cv, "rank_5pct": int(rank),
        "beta": B, "n_series": int(n), "T": int(T), "lags": lags,
        "method": "Johansen reduced-rank trace test (Osterwald-Lenum 1992 CVs)",
    }


def vecm_fit(Y, rank=1, lags=1):
    r"""Vector error-correction model at a fixed cointegrating rank.

    .. math:: \Delta Y_t = \alpha\beta' Y_{t-1}
              + \sum_i \Gamma_i \Delta Y_{t-i} + \epsilon_t

    beta comes from the Johansen eigenvectors; alpha and the Gamma
    matrices are then OLS given beta. The adjustment speeds in alpha
    are the economically interesting part: they say which series moves
    to close a disequilibrium and which is weakly exogenous.
    """
    Y = np.asarray(Y, dtype=float)
    joh = johansen(Y, lags=lags)
    n = joh["n_series"]
    r = int(rank)
    if not 1 <= r <= n:
        raise ValueError(f"rank must lie in 1..{n}, got {r}.")

    beta = joh["beta"][:, :r]
    dY = np.diff(Y, axis=0)
    T = dY.shape[0] - lags
    ect = Y[lags:-1] @ beta
    cols = [ect, np.ones((T, 1))]
    for i in range(1, lags + 1):
        cols.append(dY[lags - i : -i])
    X = np.column_stack(cols)
    y = dY[lags:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    return {
        "alpha": coef[:r].T, "beta": beta, "intercept": coef[r],
        "gamma": [coef[r + 1 + i * n : r + 1 + (i + 1) * n].T for i in range(lags)],
        "residuals": resid, "sigma": resid.T @ resid / max(T - X.shape[1], 1),
        "ect": ect, "rank": r, "lags": lags, "T": int(T),
        "eigenvalues": joh["eigenvalues"], "johansen_rank_5pct": joh["rank_5pct"],
        "method": "VECM: Johansen beta, OLS alpha and Gamma given beta",
    }


def cheatsheet():
    return "_coint: ADF + Engle-Granger (MacKinnon CVs) + Johansen trace + VECM"
