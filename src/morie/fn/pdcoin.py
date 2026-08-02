# morie.fn -- function file (rootcoder007/morie)
"""Residual-based panel cointegration statistics."""

from __future__ import annotations

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["pedroni_panel_cointegration"]

# Pedroni (1999) Table 2, "Adjustment Terms for Panel Cointegration Tests",
# transcribed from the author's working paper at
# https://web.williams.edu/Economics/wp/pedronicriticalvalues.pdf
#
# Keyed by deterministic case, then by m, the number of regressors
# excluding any constant or trend. Each entry is (mean, variance) for
# the five statistics the table covers. The table runs m = 2..7; there
# is no m = 1 row, so a bivariate regression cannot be standardised
# from it and is refused rather than extrapolated.
_PEDRONI_T2 = {
    "standard": {
        2: {"panel_v": (6.982, 81.145), "panel_rho": (-6.388, 64.288), "panel_t": (-1.662, 1.559), "group_rho": (-9.889, 41.943), "group_t": (-1.992, 0.649)},
        3: {"panel_v": (10.402, 140.804), "panel_rho": (-10.191, 89.962), "panel_t": (-2.156, 1.286), "group_rho": (-13.865, 57.801), "group_t": (-2.440, 0.600)},
        4: {"panel_v": (14.254, 182.450), "panel_rho": (-14.136, 103.176), "panel_t": (-2.571, 1.028), "group_rho": (-17.834, 72.097), "group_t": (-2.819, 0.567)},
        5: {"panel_v": (18.198, 217.784), "panel_rho": (-18.042, 120.787), "panel_t": (-2.926, 0.928), "group_rho": (-21.805, 88.611), "group_t": (-3.151, 0.559)},
        6: {"panel_v": (22.169, 256.530), "panel_rho": (-21.985, 132.499), "panel_t": (-3.244, 0.820), "group_rho": (-25.750, 103.371), "group_t": (-3.450, 0.544)},
        7: {"panel_v": (26.120, 277.429), "panel_rho": (-25.889, 143.561), "panel_t": (-3.533, 0.750), "group_rho": (-29.627, 117.059), "group_t": (-3.723, 0.530)},
    },
    "intercept": {
        2: {"panel_v": (11.754, 104.546), "panel_rho": (-9.495, 57.610), "panel_t": (-2.177, 0.964), "group_rho": (-12.938, 51.490), "group_t": (-2.453, 0.618)},
        3: {"panel_v": (15.197, 151.094), "panel_rho": (-13.256, 81.772), "panel_t": (-2.576, 0.923), "group_rho": (-16.888, 67.123), "group_t": (-2.827, 0.585)},
        4: {"panel_v": (18.910, 190.661), "panel_rho": (-17.163, 99.331), "panel_t": (-2.930, 0.843), "group_rho": (-20.841, 81.835), "group_t": (-3.157, 0.560)},
        5: {"panel_v": (22.715, 231.864), "panel_rho": (-21.013, 119.546), "panel_t": (-3.241, 0.800), "group_rho": (-24.775, 98.278), "group_t": (-3.452, 0.553)},
        6: {"panel_v": (26.603, 270.451), "panel_rho": (-24.944, 134.341), "panel_t": (-3.531, 0.750), "group_rho": (-28.720, 113.131), "group_t": (-3.726, 0.542)},
        7: {"panel_v": (30.457, 293.431), "panel_rho": (-28.795, 144.615), "panel_t": (-3.795, 0.685), "group_rho": (-32.538, 126.059), "group_t": (-3.976, 0.525)},
    },
    "trend": {
        2: {"panel_v": (21.162, 160.249), "panel_rho": (-14.011, 64.219), "panel_t": (-2.648, 0.690), "group_rho": (-17.359, 66.387), "group_t": (-2.872, 0.555)},
        3: {"panel_v": (24.556, 198.167), "panel_rho": (-17.600, 83.815), "panel_t": (-2.967, 0.686), "group_rho": (-21.116, 81.832), "group_t": (-3.179, 0.548)},
        4: {"panel_v": (28.046, 239.425), "panel_rho": (-21.287, 103.905), "panel_t": (-3.262, 0.688), "group_rho": (-24.930, 97.362), "group_t": (-3.464, 0.543)},
        5: {"panel_v": (31.738, 276.997), "panel_rho": (-25.130, 124.613), "panel_t": (-3.545, 0.686), "group_rho": (-28.849, 113.145), "group_t": (-3.737, 0.538)},
        6: {"panel_v": (35.537, 310.982), "panel_rho": (-28.981, 138.227), "panel_t": (-3.806, 0.654), "group_rho": (-32.716, 127.989), "group_t": (-3.986, 0.530)},
        7: {"panel_v": (39.231, 348.217), "panel_rho": (-32.756, 154.378), "panel_t": (-4.047, 0.638), "group_rho": (-36.494, 140.756), "group_t": (-4.217, 0.518)},
    },
}



def _newey_west_lrv(u, bandwidth=None):
    r"""Long-run variance of ``u`` by the Newey-West (1987) estimator.

    .. math:: \hat\sigma^2 = \hat\gamma_0
              + 2\sum_{s=1}^{K}\left(1 - \frac{s}{K+1}\right)\hat\gamma_s

    The default bandwidth is Newey and West's ``4 (T/100)^{2/9}``.
    """
    u = np.asarray(u, dtype=float)
    T = u.size
    if T < 2:
        return float(np.var(u)) if T else 0.0
    K = int(4 * (T / 100.0) ** (2.0 / 9.0)) if bandwidth is None else int(bandwidth)
    K = max(0, min(K, T - 1))
    uc = u - u.mean()
    g0 = float(uc @ uc) / T
    s = g0
    for lag in range(1, K + 1):
        gl = float(uc[lag:] @ uc[:-lag]) / T
        s += 2.0 * (1.0 - lag / (K + 1.0)) * gl
    return max(s, 1e-12)


def _pedroni_nuisance(y, Zx, e, bandwidth):
    """Pedroni's step 2-4 nuisance terms for one panel unit.

    Returns ``(L11_sq, lambda_i, sigma2_i)``: the long-run variance of
    the differenced-regression residuals, the Phillips-Perron style
    correction, and the long-run variance of the AR(1) residuals.
    """
    # Step 2-3: regress the differenced dependent on the differenced
    # regressors and take the long-run variance of those residuals.
    dy = np.diff(y)
    dX = np.diff(Zx, axis=0)
    if dX.size and dX.shape[0] == dy.size and dX.shape[1] > 0:
        b, *_ = np.linalg.lstsq(dX, dy, rcond=None)
        eta = dy - dX @ b
    else:
        eta = dy
    L11_sq = _newey_west_lrv(eta, bandwidth)

    # Step 4(a): e_t = gamma e_{t-1} + u_t, then sigma2 (long-run) and
    # s2 (simple) of u give lambda = (sigma2 - s2) / 2.
    lag_e = e[:-1]
    de = np.diff(e)
    denom = float(lag_e @ lag_e)
    gamma = float(lag_e @ de) / denom if denom > 0 else 0.0
    u = de - gamma * lag_e
    sigma2 = _newey_west_lrv(u, bandwidth)
    s2 = float(np.var(u))
    return L11_sq, 0.5 * (sigma2 - s2), sigma2


def _ols_resid(y, Z):
    """Residuals from regressing y on Z with an intercept."""
    D = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return y - D @ beta


def _adf_t(e, lags):
    r"""t-statistic on rho in \Delta e_t = rho e_{t-1} + sum psi_j \Delta e_{t-j}."""
    de = np.diff(e)
    n = de.size
    if n <= lags + 1:
        return np.nan
    y = de[lags:]
    cols = [e[lags:-1]]
    for j in range(1, lags + 1):
        cols.append(de[lags - j : -j] if j < lags + 1 else de[: n - lags])
    Z = np.column_stack(cols)[: y.size]
    y = y[: Z.shape[0]]
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    dof = y.size - Z.shape[1]
    if dof <= 0:
        return np.nan
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    seb = np.sqrt(s2 * XtX_inv[0, 0])
    return float(beta[0] / seb) if seb > 0 else np.nan


def pedroni_panel_cointegration(X, groups, cdf=None, lags=1, nsim=0, seed=None, case="intercept", bandwidth=None):
    r"""Residual-based panel cointegration statistics.

    Runs a cointegrating regression within each unit of the panel, then
    tests the residuals for a unit root. Under the null of no
    cointegration the residuals are I(1); under the alternative they are
    stationary. Two poolings are reported, following the within/between
    distinction Pedroni draws:

    - **panel** ("within"): the numerator and denominator are pooled
      across units before dividing, so all units share one autoregressive
      root.
    - **group** ("between"): a statistic is formed per unit and then
      averaged, so each unit keeps its own root.

    The statistics returned are the pooled and averaged
    :math:`\rho`-statistic and ADF :math:`t`-statistic.

    All five of Pedroni's tabulated statistics are computed and
    standardised, following his own five-step recipe: fit the
    cointegrating regression and keep the residuals; regress the
    differenced dependent on the differenced regressors and take the
    long-run variance of *those* residuals as :math:`\hat L_{11i}^2`;
    fit :math:`\hat e_{i,t} = \hat\gamma_i \hat e_{i,t-1} + \hat u_{i,t}`
    and form :math:`\hat\lambda_i = \tfrac12(\hat\sigma_i^2 - \hat s_i^2)`
    from the long-run and simple variances of :math:`\hat u`. Long-run
    variances use the Newey-West estimator.

    The statistics are then his Table 1 forms, for instance

    .. math::

        Z_{\hat\rho} = T\sqrt{N}\,
          \frac{\sum_i \sum_t \hat L_{11i}^{-2}
                 (\hat e_{i,t-1}\Delta\hat e_{i,t} - \hat\lambda_i)}
                {\sum_i \sum_t \hat L_{11i}^{-2} \hat e_{i,t-1}^2}

    and each is read against a normal only after his equation (2),

    .. math:: Z = \frac{\chi_{N,T} - \mu\sqrt{N}}{\sqrt{v}}

    with :math:`\mu` and :math:`v` from Table 2, transcribed here in full
    for all three deterministic cases and m = 2..7 regressors.

    The panel variance statistic diverges to :math:`+\infty` under the
    alternative, so it alone is read from the right tail; the other four
    go negative under cointegration and are read from the left.

    Table 2 has no m = 1 row, so a bivariate cointegrating regression
    cannot be standardised from it. That case is refused rather than
    extrapolated; use ``nsim`` there instead.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Panel observations. The first column is the dependent variable,
        the rest are regressors. k >= 2.
    groups : array-like, shape (n,)
        Unit label per row. At least two units, each with enough
        observations to difference and lag.
    cdf : callable, optional
        Null CDF for the group ADF statistic, replacing the standardised
        normal.
    lags : int, default 1
        Augmentation lags in the residual ADF regressions.
    nsim : int, default 0
        If positive, simulate this many independent-random-walk panels of
        the same shape and report a Monte Carlo p-value for the panel ADF
        statistic.
    seed : int, optional
        Seed for the simulation.
    bandwidth : int, optional
        Newey-West truncation lag for the long-run variances. Defaults
        to Newey and West's ``4 (T/100)^(2/9)``.
    case : {"intercept", "standard", "trend"}, default "intercept"
        Deterministic specification, selecting the block of Table 2. The
        cointegrating regression fitted here carries an intercept, so
        "intercept" is the matching default; "standard" has no
        deterministic term and "trend" adds a linear trend.

    Returns
    -------
    RichResult
        keys: ``panel_rho``, ``panel_adf``, ``group_rho``, ``group_adf``,
        ``per_unit_adf``, ``n_units``, ``p_value`` (None unless ``nsim``
        or ``cdf`` is given), ``nsim``, ``method``, ``warnings``.

    References
    ----------
    Pedroni, P. (1999). Critical values for cointegration tests in
    heterogeneous panels with multiple regressors. *Oxford Bulletin of
    Economics and Statistics*, 61(S1), 653-670.

    Pedroni, P. (2004). Panel cointegration: asymptotic and finite sample
    properties of pooled time series tests with an application to the PPP
    hypothesis. *Econometric Theory*, 20(3), 597-625.
    """
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    if Xa.ndim != 2 or Xa.shape[1] < 2:
        raise ValueError(f"X needs a dependent column and at least one regressor; got shape {Xa.shape}.")
    g = np.asarray(groups).ravel()
    if g.size != Xa.shape[0]:
        raise ValueError(f"groups must have one entry per row of X; got {g.size} and {Xa.shape[0]}.")
    if not np.all(np.isfinite(Xa)):
        raise ValueError("X must be finite.")
    units = np.unique(g)
    if units.size < 2:
        raise ValueError(f"Need at least 2 panel units, got {units.size}.")
    lags = int(lags)
    if lags < 0:
        raise ValueError(f"lags must not be negative, got {lags}.")

    # Pedroni's steps 1-4, per unit.
    A_num = A_den = 0.0     # panel (within) sums, weighted by L11^-2
    g_rho_sum = g_t_sum = 0.0
    sig_over_L = []
    rho_i = []
    adf_i = []
    skipped = []
    T_used = []
    for u in units:
        sel = g == u
        yv = Xa[sel, 0]
        Zx = Xa[sel, 1:]
        e = _ols_resid(yv, Zx)
        if e.size < 4 * (lags + 1):
            skipped.append(u)
            continue
        L11_sq, lam, sigma2 = _pedroni_nuisance(yv, Zx, e, bandwidth)
        lag_e = e[:-1]
        de = np.diff(e)
        ss = float(lag_e @ lag_e)
        if ss <= 0 or L11_sq <= 0:
            skipped.append(u)
            continue
        cross = float(lag_e @ de) - lag_e.size * lam

        # Panel (within): pooled numerator and denominator, L11^-2 weighted.
        A_den += ss / L11_sq
        A_num += cross / L11_sq
        sig_over_L.append(sigma2 / L11_sq)

        # Group (between): one ratio per unit, then summed.
        g_rho_sum += cross / ss
        g_t_sum += cross / np.sqrt(sigma2 * ss)

        rho_i.append(float(lag_e @ de) / ss)
        adf_i.append(_adf_t(e, lags))
        T_used.append(e.size)

    if A_den <= 0 or not adf_i:
        raise ValueError("No panel unit had enough usable observations; check group sizes and lags.")

    N = len(T_used)
    T_bar = float(np.mean(T_used))
    sigma_tilde2 = float(np.mean(sig_over_L))

    adf_arr = np.asarray(adf_i, dtype=float)
    finite = adf_arr[np.isfinite(adf_arr)]
    panel_adf = float(finite.mean() * np.sqrt(finite.size)) if finite.size else np.nan

    # Table 1 forms.
    stats_raw = {
        "panel_v": (T_bar**2) * (N ** 1.5) / A_den,
        "panel_rho": T_bar * np.sqrt(N) * A_num / A_den,
        "panel_t": A_num / np.sqrt(sigma_tilde2 * A_den),
        "group_rho": T_bar * (N ** -0.5) * g_rho_sum,
        "group_t": (N ** -0.5) * g_t_sum,
    }

    warn = []
    if skipped:
        warn.append(f"{len(skipped)} unit(s) skipped for too few observations: {list(skipped)}")

    m = Xa.shape[1] - 1  # regressors, excluding the intercept
    if case not in _PEDRONI_T2:
        raise ValueError(f"case must be one of {sorted(_PEDRONI_T2)}, got {case!r}.")

    # Standardise every statistic with its own Table 2 column, via
    # Pedroni eq. (2): Z = (stat - mu sqrt(N)) / sqrt(v). The panel
    # variance statistic diverges to +infinity under the alternative, so
    # it is the one read from the right tail; the other four go negative
    # under cointegration and are read from the left.
    z = {}
    pvals = {}
    if m in _PEDRONI_T2[case]:
        for name, raw in stats_raw.items():
            mu, v = _PEDRONI_T2[case][m][name]
            zz = (raw - mu * np.sqrt(N)) / np.sqrt(v)
            z[name] = float(zz)
            pvals[name] = float(stats.norm.sf(zz)) if name == "panel_v" else float(stats.norm.cdf(zz))
    else:
        warn.append(
            f"Pedroni Table 2 covers m = 2..7 regressors; this panel has m = {m}, "
            "so no standardised p-values are available. Use nsim for a simulated null."
        )

    p = pvals.get("group_t")
    if cdf is not None:
        p = float(cdf(panel_adf))
    elif p is None and nsim and nsim > 0:
        rng = np.random.default_rng(seed)
        nsim = int(nsim)
        null = np.empty(nsim)
        for b in range(nsim):
            sim = np.empty_like(Xa)
            for u in units:
                sel = g == u
                mm = int(np.sum(sel))
                sim[sel] = np.cumsum(rng.normal(0, 1, (mm, Xa.shape[1])), axis=0)
            null[b] = pedroni_panel_cointegration(sim, g, lags=lags, case=case)["group_adf"]
        good = null[np.isfinite(null)]
        p = (1.0 + float(np.sum(good <= panel_adf))) / (1.0 + good.size) if good.size else None

    warn.append(
        "The two ADF (parametric) statistics use Table 2's t columns, which Pedroni "
        "tabulates jointly for the parametric and non-parametric t forms."
    )

    return RichResult(
        title="Panel cointegration (residual-based)",
        payload={
            "statistics": stats_raw,
            "z": z,
            "p_values": pvals,
            "panel_v": stats_raw["panel_v"],
            "panel_rho": stats_raw["panel_rho"],
            "panel_t": stats_raw["panel_t"],
            "group_rho": stats_raw["group_rho"],
            "group_t": stats_raw["group_t"],
            "group_adf": panel_adf,
            "mean_adf_t": float(finite.mean()) if finite.size else np.nan,
            "per_unit_adf": adf_arr,
            "n_units": int(N),
            "n_regressors": int(m),
            "case": case,
            "p_value": p,
            "nsim": int(nsim),
            "method": "Pedroni residual-based panel cointegration; group ADF standardised by Table 2",
            "warnings": warn,
        },
    )


def cheatsheet():
    return "pdcoin: residual-based panel cointegration statistics"
