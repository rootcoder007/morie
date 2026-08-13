# morie.fn -- function file (rootcoder007/morie)
r"""N-BEATS as an ensemble forecaster.

Same model and same source as :mod:`morie.fn.nbeats` -- the ledger
carries both rows against Oreshkin et al. (2020), and they describe one
architecture, so this module does not re-derive the doubly residual
stack. What it adds is the part of the paper that is *not* the
architecture and that the single-model row does not cover: **the
ensemble is the method**.

**The reported results are an ensemble, not a model.** The paper's
headline numbers come from averaging 180 models -- across three loss
functions, six lookback multiples of the horizon, and ten
initialisations. Reporting a single N-BEATS fit and citing those
numbers compares different things. The ensemble members differ in ways
chosen to *decorrelate their errors*, which is why the aggregate beats
every member.

**The median is the aggregator, not the mean.** A single member that
diverges drags a mean with it; the median ignores it. The anchor plants
an outlier member and checks the median ensemble absorbs it while the
mean does not -- which is the difference between the two, measured.

**Multiple lookbacks are the largest source of diversity.** Members
trained on different history lengths see genuinely different problems:
a short lookback tracks recent level, a long one sees the seasonal
shape. That is a bigger effect than reinitialisation, and the anchor
measures the spread each source contributes rather than assuming it.

References
----------
Oreshkin, B. N., Carpov, D., Chapados, N. & Bengio, Y. (2020)
"N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting", *International Conference on Learning Representations*,
arXiv:1905.10437. Sec. 3.3 (ensembling), Sec. 5.

Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020) "The M4
Competition: 100,000 time series and 61 forecasting methods",
*International Journal of Forecasting* 36(1), 54-74,
doi:10.1016/j.ijforecast.2019.04.014. The benchmark those numbers come
from, and where combination dominated.

Bates, J. M. & Granger, C. W. J. (1969) "The Combination of Forecasts",
*Journal of the Operational Research Society* 20(4), 451-468,
doi:10.1057/jors.1969.103. Why combining decorrelated forecasts beats
choosing among them.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .nbeats import nbeats_stack

__all__ = ["ensemble_members", "aggregate_forecasts",
           "nbeats_ensemble"]

_EPS = 1e-12
_AGG = ("median", "mean")


def ensemble_members(y, horizon, lookback_multiples=(2, 3, 4, 5, 6, 7),
                     block_sets=None, ridge=1e-8):
    r"""One member per (lookback, block configuration).

    The lookback sweep is the paper's largest diversity source: a short
    history tracks the recent level, a long one sees the seasonal
    shape, and those are genuinely different problems.
    """
    yv = k.vec(y)
    n = len(yv)
    H = int(horizon)
    sets = ([[("trend", 2, 3), ("seasonality", 2, 3)],
             [("trend", 1, 3), ("seasonality", 3, 3)],
             [("generic", 0, 0), ("trend", 2, 3)]]
            if block_sets is None else list(block_sets))
    out = []
    for mult in lookback_multiples:
        lb = int(mult) * H
        if lb < 4 or lb > n:
            continue
        window = yv[n - lb:]
        for si, blocks in enumerate(sets):
            try:
                fc, resid, _ = nbeats_stack(window, H, blocks,
                                            ridge=ridge)
            except (ValueError, ZeroDivisionError):
                continue
            out.append({"lookback": lb, "multiple": int(mult),
                        "block_set": si, "forecast": fc,
                        "residual_norm": math.sqrt(sum(v * v
                                                       for v in resid))})
    if not out:
        raise ValueError("ngnest: no ensemble member could be built; "
                         "the series is too short for these lookbacks")
    return out


def aggregate_forecasts(members, how="median"):
    r"""Combine member forecasts pointwise.

    The median is the paper's choice and it is not cosmetic: one
    diverging member drags a mean and is ignored by a median.
    """
    if how not in _AGG:
        raise ValueError("ngnest: how must be median or mean, got %r"
                         % (how,))
    if not members:
        raise ValueError("ngnest: no members to aggregate")
    H = len(members[0]["forecast"])
    if any(len(m["forecast"]) != H for m in members):
        raise ValueError("ngnest: members disagree on the horizon")
    out = []
    for h in range(H):
        col = [m["forecast"][h] for m in members]
        out.append(k.median(col) if how == "median"
                   else sum(col) / len(col))
    return out


def nbeats_ensemble(y, horizon, lookback_multiples=(2, 3, 4, 5, 6, 7),
                    block_sets=None, how="median", ridge=1e-8):
    r"""The ensemble forecast, with the spread its members produced."""
    mem = ensemble_members(y, horizon, lookback_multiples, block_sets,
                           ridge)
    agg = aggregate_forecasts(mem, how=how)
    H = len(agg)
    spread = [max(m["forecast"][h] for m in mem)
              - min(m["forecast"][h] for m in mem) for h in range(H)]
    by_lb = {}
    for m in mem:
        by_lb.setdefault(m["multiple"], []).append(m["forecast"][0])
    by_set = {}
    for m in mem:
        by_set.setdefault(m["block_set"], []).append(m["forecast"][0])
    return RichResult(payload={
        "estimate": agg, "forecast": agg, "members": mem,
        "n_members": len(mem), "spread": spread,
        "aggregator": how,
        "mean_forecast": aggregate_forecasts(mem, how="mean"),
        "lookback_spread": (max(k.mean(v) for v in by_lb.values())
                            - min(k.mean(v) for v in by_lb.values())),
        "blockset_spread": (max(k.mean(v) for v in by_set.values())
                            - min(k.mean(v) for v in by_set.values())),
        "lookbacks": sorted(by_lb), "n_block_sets": len(by_set),
        "method": "N-BEATS ensemble over lookbacks and block "
                  "configurations, Oreshkin et al. (2020) Sec. 3.3",
    })


def cheatsheet():
    return ("ngnest: same source as nbeats -- this is the ENSEMBLE, "
            "which is what the paper's numbers actually are (180 models "
            "over losses, lookbacks and seeds). Aggregate by MEDIAN, "
            "not mean: one diverging member drags a mean and is ignored "
            "by a median. Varying the lookback decorrelates members "
            "more than reinitialising does.")


# compact alias per ledger/NAMING.md
nbeatsensemble = nbeats_ensemble
