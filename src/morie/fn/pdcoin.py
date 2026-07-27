# morie.fn -- function file (rootcoder007/morie)
"""Residual-based panel cointegration statistics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["pedroni_panel_cointegration"]


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


def pedroni_panel_cointegration(X, groups, cdf=None, lags=1, nsim=0, seed=None):
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

    **What this does not do.** Pedroni's published tests standardise each
    raw statistic by tabulated moments, ``(stat - mu*sqrt(N)) /
    sqrt(v)``, with :math:`\mu` and :math:`v` read from his simulation
    tables, and only then compare against a standard normal. Those tables
    are not reproduced here, so no asymptotic p-value is reported by
    default: an unstandardised statistic compared against a normal would
    be wrong, quietly. Set ``nsim`` to obtain a p-value by simulating the
    null -- independent random walks of the same shape -- which needs no
    tables, or pass ``cdf`` if you have the appropriate null in hand.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Panel observations. The first column is the dependent variable,
        the rest are regressors. k >= 2.
    groups : array-like, shape (n,)
        Unit label per row. At least two units, each with enough
        observations to difference and lag.
    cdf : callable, optional
        Null CDF for the panel ADF statistic.
    lags : int, default 1
        Augmentation lags in the residual ADF regressions.
    nsim : int, default 0
        If positive, simulate this many independent-random-walk panels of
        the same shape and report a Monte Carlo p-value for the panel ADF
        statistic.
    seed : int, optional
        Seed for the simulation.

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
    warn.append(
        "Statistics are unstandardised: Pedroni's tabulated mu and v are not applied, "
        "so they are not directly comparable with published critical values."
    )

    p = None
    if cdf is not None:
        p = float(cdf(panel_adf))
    elif nsim and nsim > 0:
        rng = np.random.default_rng(seed)
        nsim = int(nsim)
        null = np.empty(nsim)
        for b in range(nsim):
            sim = np.empty_like(Xa)
            for u in units:
                sel = g == u
                m = int(np.sum(sel))
                sim[sel] = np.cumsum(rng.normal(0, 1, (m, Xa.shape[1])), axis=0)
            null[b] = pedroni_panel_cointegration(sim, g, lags=lags)["panel_adf"]
        good = null[np.isfinite(null)]
        p = (1.0 + float(np.sum(good <= panel_adf))) / (1.0 + good.size) if good.size else None

    return RichResult(
        title="Panel cointegration (residual-based)",
        payload={
            "panel_rho": num / den,
            "panel_adf": panel_adf,
            "group_rho": float(np.mean(rho_i)) if rho_i else np.nan,
            "group_adf": float(finite.mean()) if finite.size else np.nan,
            "per_unit_adf": adf_arr,
            "n_units": int(units.size - len(skipped)),
            "p_value": p,
            "nsim": int(nsim),
            "method": "Residual-based panel cointegration, within and between poolings",
            "warnings": warn,
        },
    )


def cheatsheet():
    return "pdcoin: residual-based panel cointegration statistics"
