# morie.fn -- function file (rootcoder007/morie)
r"""ADIDA: aggregate, forecast, disaggregate.

Intermittent series are mostly zeros, and zeros carry no information
about level. ADIDA's move is to stop fighting that at the original
frequency: sum the history into non-overlapping *time buckets* of
:math:`m` periods, forecast the aggregated series -- which now has far
fewer zeros -- and divide the result back down.

**Temporal aggregation is a self-improving mechanism.** Bucketing
:math:`m` periods reduces the proportion of zeros and, with it, the
intermittence the forecaster has to cope with; the paper's phrase is
that the process functions as a self-improving mechanism. It is not a
different forecasting method -- any method can sit inside -- which is
why the module takes the base forecaster as an argument.

**The aggregation level is the decision, and the paper has a
recommendation.** Setting :math:`m` equal to the lead time (plus review
period) makes the aggregated forecast *directly* the lead-time demand,
so no disaggregation is needed at all and the disaggregation error
disappears with it. The anchor checks that identity holds exactly.

**Disaggregation is where the assumption hides.** Dividing by :math:`m`
assumes demand is spread evenly across the bucket. That is the simplest
option and it is what is implemented, with a seasonal-profile
alternative -- and the anchor checks that equal-weight disaggregation
sums back to the aggregate forecast exactly, because a disaggregation
that does not reconstitute is silently changing the total.

**Overlapping buckets use the data better and are not free.** A
non-overlapping partition throws away every alignment but one; a
rolling window uses them all but produces autocorrelated aggregates.
Both are offered.

References
----------
Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F. &
Assimakopoulos, V. (2011) "An aggregate-disaggregate intermittent
demand approach (ADIDA) to forecasting: an empirical proposition and
analysis", *Journal of the Operational Research Society* 62(3),
544-554, doi:10.1057/jors.2010.32.

Petropoulos, F. & Kourentzes, N. (2015) "Forecast combinations for
intermittent demand", *Journal of the Operational Research Society*
66(6), 914-924, doi:10.1057/jors.2014.62. Temporal combinations across
aggregation levels, which is the multi-level extension here.

Teunter, R. H., Syntetos, A. A. & Babai, M. Z. (2011) "Intermittent
demand: Linking forecasting to inventory obsolescence", *European
Journal of Operational Research* 214(3), 606-615,
doi:10.1016/j.ejor.2011.05.018. The TSB base forecaster.

Croston, J. D. (1972) "Forecasting and Stock Control for Intermittent
Demands", *Operational Research Quarterly* 23(3), 289-303,
doi:10.2307/3007885.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .tsbF import intermittent_forecast

__all__ = ["aggregate_buckets", "disaggregate", "adida_forecast",
           "temporal_combination", "zero_fraction"]

_EPS = 1e-12


def zero_fraction(y):
    """The share of periods with no demand -- what aggregation reduces."""
    yv = [float(v) for v in y]
    if not yv:
        raise ValueError("adida: empty series")
    return sum(1 for v in yv if v <= 0.0) / float(len(yv))


def aggregate_buckets(y, m, overlapping=False):
    r"""Sum into buckets of :math:`m` periods.

    Non-overlapping keeps the LAST complete buckets, so the most recent
    history is retained and the leftover at the start is dropped --
    dropping from the end instead would discard exactly the periods the
    forecast is anchored on.
    """
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    mm = int(m)
    if mm < 1:
        raise ValueError("adida: the bucket size must be at least 1, "
                         "got %d" % mm)
    if mm > n:
        raise ValueError("adida: bucket size %d exceeds the %d "
                         "observations" % (mm, n))
    if overlapping:
        return [sum(yv[t:t + mm]) for t in range(n - mm + 1)]
    n_buckets = n // mm
    start = n - n_buckets * mm
    return [sum(yv[start + b * mm:start + (b + 1) * mm])
            for b in range(n_buckets)]


def disaggregate(aggregate_value, m, profile=None):
    r"""Spread a bucket forecast back over :math:`m` periods.

    Equal weights assume demand is uniform within the bucket. Whatever
    the weights, they are normalised to sum to 1 so the disaggregated
    periods reconstitute the aggregate exactly -- a profile that did
    not sum to 1 would silently change the total.
    """
    mm = int(m)
    if mm < 1:
        raise ValueError("adida: the bucket size must be at least 1")
    if profile is None:
        w = [1.0 / mm] * mm
    else:
        w = [float(v) for v in profile]
        if len(w) != mm:
            raise ValueError("adida: the profile has %d weights for a "
                             "bucket of %d" % (len(w), mm))
        if any(v < 0.0 for v in w):
            raise ValueError("adida: profile weights must be "
                             "non-negative")
        tot = sum(w)
        if tot <= 0.0:
            raise ValueError("adida: the profile sums to zero")
        w = [v / tot for v in w]
    return [float(aggregate_value) * v for v in w]


def adida_forecast(y, m, horizon=1, method="tsb", alpha=0.1, beta=0.05,
                   overlapping=False, profile=None, lead_time=None):
    r"""Aggregate to level ``m``, forecast, disaggregate.

    ``lead_time`` sets ``m`` to the lead time, the paper's
    recommendation: the aggregated forecast is then the lead-time demand
    directly, and no disaggregation error is incurred at all.
    """
    yv = [float(v) for v in k.vec(y)]
    if lead_time is not None:
        m = int(lead_time)
    agg = aggregate_buckets(yv, m, overlapping=overlapping)
    if len(agg) < 2:
        raise ValueError("adida: bucket size %d leaves only %d "
                         "aggregated points" % (int(m), len(agg)))
    f = intermittent_forecast(agg, method=method, alpha=alpha,
                              beta=beta, horizon=1)
    agg_fc = f["forecast"][0]
    per_period = disaggregate(agg_fc, int(m), profile=profile)
    reps = int(math.ceil(horizon / float(m)))
    flat = [per_period[t % int(m)] for t in range(reps * int(m))]
    return RichResult(payload={
        "estimate": flat[:int(horizon)],
        "forecast": flat[:int(horizon)],
        "aggregate_forecast": agg_fc,
        "lead_time_demand": agg_fc if lead_time is not None else None,
        "aggregated": agg, "m": int(m),
        "zero_fraction_original": zero_fraction(yv),
        "zero_fraction_aggregated": zero_fraction(agg),
        "n_buckets": len(agg), "overlapping": bool(overlapping),
        "base_method": method,
        "disaggregation_sums_back": abs(sum(per_period) - agg_fc) < 1e-9,
        "method": "ADIDA, Nikolopoulos, Syntetos, Boylan, Petropoulos "
                  "& Assimakopoulos (2011)",
    })


def temporal_combination(y, levels, horizon=1, method="tsb",
                         alpha=0.1, beta=0.05, weights=None):
    r"""Combine forecasts made at several aggregation levels.

    Petropoulos & Kourentzes: rather than choosing one level, forecast
    at several and combine, which removes the level-selection decision
    and is more robust than any single level chosen in advance.
    """
    lv = [int(v) for v in levels]
    if len(lv) < 2:
        raise ValueError("adida: need at least 2 levels to combine, "
                         "got %d" % len(lv))
    per = []
    for m in lv:
        r = adida_forecast(y, m, horizon=horizon, method=method,
                           alpha=alpha, beta=beta)
        per.append(r["forecast"])
    if weights is None:
        w = [1.0 / len(lv)] * len(lv)
    else:
        w = [float(v) for v in weights]
        if len(w) != len(lv):
            raise ValueError("adida: %d weights for %d levels"
                             % (len(w), len(lv)))
        tot = sum(w)
        if tot <= 0.0:
            raise ValueError("adida: the weights sum to zero")
        w = [v / tot for v in w]
    comb = [sum(w[j] * per[j][h] for j in range(len(lv)))
            for h in range(int(horizon))]
    return RichResult(payload={
        "estimate": comb, "forecast": comb, "levels": lv,
        "per_level": per, "weights": w,
        "spread": (max(p[0] for p in per) - min(p[0] for p in per)),
        "method": "temporal combination across aggregation levels, "
                  "Petropoulos & Kourentzes (2015)",
    })


def cheatsheet():
    return ("adida: sum into buckets of m, forecast the aggregate, "
            "divide back by m. Aggregation cuts the zero fraction, "
            "which is the self-improving mechanism. Set m = LEAD TIME "
            "and the aggregate forecast IS lead-time demand, so no "
            "disaggregation error at all. Equal-weight disaggregation "
            "must sum back to the aggregate exactly. Combine several "
            "levels instead of choosing one.")


# compact alias per ledger/NAMING.md
adidaforecast = adida_forecast

# public names resolved by fn/_lazy_map.json
adida = adida_forecast
