# morie.fn -- function file (rootcoder007/morie)
"""Multi-horizon distributional accuracy test (Corradi-Swanson type)."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["vol_corradi_swan_persistence"]


def _ks_stat(sample, cdf_vals):
    n = sample.size
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(ecdf_hi - cdf_vals), np.max(cdf_vals - ecdf_lo)))


def _mc_pvalue_fitted_normal(d_obs, n, n_mc, rng):
    """Null distribution of the KS statistic when mean and sd are fitted.

    Parameter-free for the location-scale Gaussian family, so simulating
    standard normals is exact (the Lilliefors construction).
    """
    count = 0
    for _ in range(n_mc):
        z = np.sort(rng.standard_normal(n))
        cdf = stats.norm.cdf(z, loc=z.mean(), scale=z.std(ddof=1))
        if _ks_stat(z, cdf) >= d_obs:
            count += 1
    return (1.0 + count) / (1.0 + n_mc)


def vol_corradi_swan_persistence(r, horizons=(1, 5, 20), cdf=None, n_mc=500, seed=0):
    r"""Distributional accuracy of a returns model across horizons.

    For each horizon h the series is aggregated into non-overlapping
    h-period sums and a Kolmogorov-type statistic

    .. math:: V_h = \sup_x |F_{n_h}(x) - F_h(x)|

    compares their empirical distribution with the model's distribution
    at that horizon -- the :math:`V_{1T}`-type comparison of Corradi &
    Swanson (2006), who evaluate predictive densities by exactly this
    kind of sup-distance on the fitted CDF. A model can fit the
    one-period distribution and still fail at 20 periods: under
    volatility persistence the aggregated returns stay fat-tailed far
    longer than an i.i.d. model predicts, which is what checking
    several horizons detects and a single-horizon test cannot.

    P-values. With ``cdf`` supplied the null is fully specified and the
    classical Kolmogorov distribution applies. With ``cdf=None`` a
    Gaussian is FITTED per horizon, the classical p-value would be
    badly conservative, and the null distribution of :math:`V_h` is
    instead simulated (the Lilliefors construction, exact for a fitted
    location-scale family). The joint p-value is Bonferroni across
    horizons.

    This replaces a placeholder that computed a single-horizon KS
    normality statistic, ignored ``horizons`` entirely, and used the
    classical p-value on fitted parameters.

    Parameters
    ----------
    r : array-like, shape (n,)
        Return series.
    horizons : sequence of int, default (1, 5, 20)
        Aggregation horizons in periods.
    cdf : callable, optional
        ``cdf(x, h)`` giving the model CDF of an h-period aggregate at
        x. When omitted, a Gaussian is fitted per horizon.
    n_mc : int, default 500
        Monte Carlo replicates for the fitted-parameter null.
    seed : int, default 0
        Seed for the Monte Carlo.

    Returns
    -------
    RichResult
        keys: ``statistic`` (max over horizons), ``p_value`` (Bonferroni
        joint), ``per_horizon`` (list of dicts with h, n_h, statistic,
        p_value), ``horizons``, ``n``, ``method``.

    References
    ----------
    Corradi, V. & Swanson, N. R. (2006). Predictive density and
    conditional confidence interval accuracy tests. *Journal of
    Econometrics*, 135(1-2), 187-228.
    Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for
    normality with mean and variance unknown. *JASA*, 62(318), 399-402
    (the fitted-parameter null by simulation).
    """
    r = np.asarray(r, dtype=float).ravel()
    n = r.size
    horizons = [int(h) for h in np.atleast_1d(horizons)]
    if any(h < 1 for h in horizons):
        raise ValueError(f"horizons must be positive, got {horizons}.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite.")
    hmax = max(horizons)
    if n < 8 * hmax:
        raise ValueError(
            f"Need at least 8 aggregates at the longest horizon; n={n} gives {n // hmax} at h={hmax}."
        )

    rng = np.random.default_rng(seed)
    per = []
    for h in horizons:
        m = n // h
        agg = np.sort(r[: m * h].reshape(m, h).sum(axis=1))
        if cdf is None:
            cdf_vals = stats.norm.cdf(agg, loc=agg.mean(), scale=agg.std(ddof=1))
            d = _ks_stat(agg, cdf_vals)
            p = _mc_pvalue_fitted_normal(d, m, n_mc, rng)
        else:
            cdf_vals = np.array([float(cdf(x, h)) for x in agg])
            d = _ks_stat(agg, cdf_vals)
            p = float(stats.kstwo.sf(d, m))
        per.append({"h": h, "n_h": int(m), "statistic": d, "p_value": float(p)})

    stat = max(e["statistic"] for e in per)
    p_joint = min(1.0, len(per) * min(e["p_value"] for e in per))
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(p_joint),
            "per_horizon": per,
            "horizons": horizons,
            "n": int(n),
            "method": "Multi-horizon KS-type distributional accuracy (Corradi-Swanson type)",
        }
    )


def cheatsheet():
    return "volcorpst: multi-horizon distributional accuracy (Corradi-Swanson type)"
