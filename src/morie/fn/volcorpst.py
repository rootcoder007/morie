# morie.fn -- function file (rootcoder007/morie)
"""Multi-horizon distributional accuracy test (Corradi-Swanson type)."""

import math

from . import _stats_core as stats
from ._richresult import RichResult
from ._rng import random_normal

__all__ = ["vol_corradi_swan_persistence"]


def _pnorm(x, m, s):
    return 0.5 * math.erfc(-(x - m) / (s * math.sqrt(2.0)))


def _ks_stat(sorted_sample, cdf_vals):
    n = len(sorted_sample)
    return max(max((i + 1) / n - c for i, c in enumerate(cdf_vals)), max(c - i / n for i, c in enumerate(cdf_vals)))


def _fitted_ks(z):
    """KS distance of a sorted sample from the Gaussian with its own mean and SD."""
    n = len(z)
    m = math.fsum(z) / n
    s = math.sqrt(math.fsum((v - m) ** 2 for v in z) / (n - 1))
    return _ks_stat(z, [_pnorm(v, m, s) for v in z])


def _mc_pvalue_fitted_normal(d_obs, n, n_mc, seed, block):
    """Null distribution of the KS statistic when mean and sd are fitted.

    Parameter-free for the location-scale Gaussian family, so simulating
    standard normals is exact (the Lilliefors construction); replicate k
    uses Philox stream ``block * n_mc + k``.
    """
    count = 0
    for k in range(n_mc):
        z = sorted(float(v) for v in random_normal(n, seed=seed, stream=block * n_mc + k))
        if _fitted_ks(z) >= d_obs:
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
        Philox seed of the Monte Carlo (stream ``i * n_mc + k`` for
        replicate ``k`` of the ``i``-th horizon, identical in the R arm).

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

    Examples
    --------
    >>> import math
    >>> r = [0.01 * math.sin(1.7 * t) + 0.004 * math.cos(0.3 * t * t) for t in range(120)]
    >>> out = vol_corradi_swan_persistence(r, horizons=(1, 4), n_mc=49)
    >>> [round(e["statistic"], 10) for e in out["per_horizon"]]
    [0.0957967024, 0.1211485613]
    """
    rv = [float(v) for v in (r.tolist() if hasattr(r, "tolist") else r)]
    n = len(rv)
    horizons = tuple(int(h) for h in horizons)
    per = []
    for i, h in enumerate(horizons):
        m = n // h
        if m < 3:
            raise ValueError(f"horizon {h} leaves fewer than 3 aggregates")
        agg = sorted(math.fsum(rv[j * h : (j + 1) * h]) for j in range(m))
        if cdf is None:
            d = _fitted_ks(agg)
            p = _mc_pvalue_fitted_normal(d, m, int(n_mc), seed, i)
        else:
            d = _ks_stat(agg, [float(cdf(x, h)) for x in agg])
            p = float(stats.kstwo.sf(d, m))
        per.append({"h": h, "n_h": m, "statistic": d, "p_value": p})
    stat = max(e["statistic"] for e in per)
    p_joint = min(1.0, len(per) * min(e["p_value"] for e in per))
    return RichResult(
        payload={
            "statistic": stat,
            "p_value": p_joint,
            "per_horizon": per,
            "horizons": horizons,
            "n": n,
            "method": "Multi-horizon KS-type distributional accuracy (Corradi-Swanson type)",
        }
    )


def cheatsheet():
    return "volcorpst: multi-horizon distributional accuracy (Corradi-Swanson type)"
