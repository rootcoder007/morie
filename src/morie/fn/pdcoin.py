# morie.fn -- function file (rootcoder007/morie)
"""Residual-based panel cointegration statistics."""

from __future__ import annotations

import numpy as np
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


def pedroni_panel_cointegration(X, groups, cdf=None, lags=1, nsim=0, seed=None, case="intercept"):
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

    **The group ADF statistic is standardised; the others are not.**
    Pedroni's tests are read against a normal only after subtracting a
    tabulated mean and dividing by a tabulated standard deviation,

    .. math:: Z = \frac{\chi_{N,T} - \mu\sqrt{N}}{\sqrt{v}}
              \;\Rightarrow\; N(0,1)

    his equation (2), with :math:`\mu` and :math:`v` from his Table 2.
    That table is transcribed here in full, for all three deterministic
    cases and m = 2..7 regressors.

    Of the statistics computed here, ``group_adf`` is already in the form
    his Table 1 defines, :math:`N^{-1/2}\sum_i t_i`, so it is
    standardised with the "Group t" column and gets a genuine asymptotic
    p-value. ``panel_rho`` and ``group_rho`` are the raw pooled and
    averaged autoregressive coefficients: they are *not* in Pedroni's
    standardised form, which requires the long-run variance corrections
    :math:`\hat\lambda_i`, :math:`\hat\sigma_i^2` and
    :math:`\hat L_{11i}` that this implementation does not compute. They
    are reported as descriptive quantities and are not given p-values,
    because applying the Table 2 terms to a statistic that is not in the
    matching form would be wrong quietly.

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

    num = 0.0
    den = 0.0
    rho_i = []
    adf_i = []
    skipped = []
    for u in units:
        sel = g == u
        e = _ols_resid(Xa[sel, 0], Xa[sel, 1:])
        if e.size < 4 * (lags + 1):
            skipped.append(u)
            continue
        lag_e = e[:-1]
        de = np.diff(e)
        num += float(lag_e @ de)
        den += float(lag_e @ lag_e)
        if lag_e @ lag_e > 0:
            rho_i.append(float(lag_e @ de) / float(lag_e @ lag_e))
        adf_i.append(_adf_t(e, lags))

    if den <= 0 or not adf_i:
        raise ValueError("No panel unit had enough usable observations; check group sizes and lags.")

    adf_arr = np.asarray(adf_i, dtype=float)
    finite = adf_arr[np.isfinite(adf_arr)]
    panel_adf = float(finite.mean() * np.sqrt(finite.size)) if finite.size else np.nan

    warn = []
    if skipped:
        warn.append(f"{len(skipped)} unit(s) skipped for too few observations: {list(skipped)}")

    m = Xa.shape[1] - 1  # regressors, excluding the intercept
    if case not in _PEDRONI_T2:
        raise ValueError(f"case must be one of {sorted(_PEDRONI_T2)}, got {case!r}.")

    z = None
    p = None
    if cdf is not None:
        p = float(cdf(panel_adf))
    elif m in _PEDRONI_T2[case]:
        # group_adf is N^-1/2 sum_i t_i, which is Pedroni's Table 1 form
        # for the group t statistic, so Table 2's "Group t" column
        # applies directly.
        mu, v = _PEDRONI_T2[case][m]["group_t"]
        z = (panel_adf - mu * np.sqrt(finite.size)) / np.sqrt(v)
        p = float(stats.norm.cdf(z))  # left tail: cointegration drives it negative
    else:
        warn.append(
            f"Pedroni Table 2 covers m = 2..7 regressors; this panel has m = {m}, "
            "so no standardised p-value is available. Use nsim for a simulated null."
        )

    if p is None and nsim and nsim > 0:
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
        "panel_rho and group_rho are raw pooled/averaged autoregressive coefficients, "
        "not Pedroni's standardised forms; they carry no p-value."
    )

    return RichResult(
        title="Panel cointegration (residual-based)",
        payload={
            "panel_rho": num / den,
            "group_adf": panel_adf,
            "group_rho": float(np.mean(rho_i)) if rho_i else np.nan,
            "mean_adf_t": float(finite.mean()) if finite.size else np.nan,
            "z_group_adf": z,
            "per_unit_adf": adf_arr,
            "n_units": int(units.size - len(skipped)),
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
