# morie.fn -- function file (rootcoder007/morie)
r"""Prophet's automatic changepoint selection.

Where does a trend change? Specifying the dates by hand needs the
answer in advance. Prophet instead puts a large number of candidate
changepoints down -- one per month over several years is typical -- and
lets a sparse prior decide which are real:

.. math:: \delta_j \sim \mathrm{Laplace}(0, \tau).

**Sparsity is the whole mechanism, and :math:`\tau` is the only dial.**
The Laplace prior is the Bayesian form of an L1 penalty, so most
:math:`\delta_j` are driven to zero and a handful survive. Small
:math:`\tau` means heavy shrinkage and a nearly straight trend; large
:math:`\tau` lets the trend bend wherever the data suggests. The paper
is explicit about the consequence: as :math:`\tau` grows the training
error falls and the *forecast* uncertainty widens, because the
flexibility that fits the history is projected forward. That trade-off
is measured here rather than described -- the anchor sweeps
:math:`\tau` and checks training error falls monotonically while the
number of active changepoints rises.

**Changepoints are placed only in the first 80% of the history.** A
changepoint near the end has almost no data after it to estimate its
rate, so it fits the last few points and then dominates every forecast.
The default ``changepoint_range=0.8`` is that guard, and it is a
default worth understanding rather than tuning away.

**Forecast uncertainty comes from projecting the same rate of change
forward.** Future changepoints are simulated at the historical
frequency :math:`S/T` with magnitudes drawn from the inferred
:math:`\mathrm{Laplace}(0,\lambda)`, so the interval widens with
horizon. The paper says plainly that this will not have exact coverage;
it is an indication, and above all an overfitting detector.

References
----------
Taylor, S. J. & Letham, B. (2018) "Forecasting at Scale", *The American
Statistician* 72(1), 37-45, doi:10.1080/00031305.2017.1380080;
preprint *PeerJ Preprints* 5:e3190v2,
doi:10.7287/peerj.preprints.3190v2. Sec. 3.1.3 (automatic changepoint
selection) and Sec. 3.1.4 (trend forecast uncertainty).

Tibshirani, R. (1996) "Regression Shrinkage and Selection via the
Lasso", *Journal of the Royal Statistical Society Series B* 58(1),
267-288, doi:10.1111/j.2517-6161.1996.tb02080.x. The L1 penalty the
Laplace prior corresponds to.

Park, T. & Casella, G. (2008) "The Bayesian Lasso", *Journal of the
American Statistical Association* 103(482), 681-686,
doi:10.1198/016214508000000337. The Laplace-prior formulation.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult
from .prphet import piecewise_trend, prophet_fit

__all__ = ["changepoint_path", "select_changepoints",
           "simulate_future_trend", "trend_intervals"]

_EPS = 1e-12


def changepoint_path(t, y, taus=None, n_changepoints=15,
                     changepoint_range=0.8, seasonalities=None, **kw):
    r"""Sweep :math:`\tau` and report what it buys and costs.

    Rising :math:`\tau` must lower the training error and raise the
    number of active changepoints -- that trade-off is the paper's
    point about overfitting, and it is measured here.
    """
    grid = ([0.001, 0.01, 0.05, 0.1, 0.5, 1.0] if taus is None
            else [float(v) for v in taus])
    if len(grid) < 2:
        raise ValueError("prnFil: need at least 2 tau values, got %d"
                         % len(grid))
    rows = []
    for tau in grid:
        f = prophet_fit(t, y, n_changepoints=n_changepoints,
                        changepoint_range=changepoint_range,
                        changepoint_prior=tau,
                        seasonalities=seasonalities, **kw)
        d = f["deltas"]
        # exactly zero, not "small": the L1 solution really does zero
        # them, which is the whole point of the Laplace prior
        active = sum(1 for v in d if v != 0.0)
        rows.append({"tau": tau, "active": active,
                     "rmse": math.sqrt(k.mean([v * v
                                               for v in f["residual"]])),
                     "l1": sum(abs(v) for v in d),
                     "deltas": d})
    return rows


def select_changepoints(t, y, tau=0.05, n_changepoints=15,
                        changepoint_range=0.8, seasonalities=None,
                        **kw):
    r"""Fit once and report which candidates survived the shrinkage.

    Survival means a delta that is EXACTLY zero has been dropped, not
    one that is merely small -- the L1 solution produces genuine zeros,
    and a magnitude threshold on a ridge fit would be a different
    method wearing the same name.
    """
    f = prophet_fit(t, y, n_changepoints=n_changepoints,
                    changepoint_range=changepoint_range,
                    changepoint_prior=tau, seasonalities=seasonalities,
                    **kw)
    d = f["deltas"]
    cps = f["changepoints"]
    keep = [j for j in range(len(d)) if d[j] != 0.0]
    tv = f["t"]
    span = tv[-1] - tv[0]
    return RichResult(payload={
        "estimate": [cps[j] for j in keep],
        "selected": [cps[j] for j in keep],
        "selected_index": keep, "deltas": d, "candidates": cps,
        "n_selected": len(keep), "n_candidates": len(cps),
        "tau": float(tau), "fit": f,
        "last_candidate_fraction": ((cps[-1] - tv[0]) / span
                                    if cps and span > 0 else 0.0),
        "changepoint_range": float(changepoint_range),
        "rmse": math.sqrt(k.mean([v * v for v in f["residual"]])),
        "method": "automatic changepoint selection under a Laplace "
                  "prior, Taylor & Letham (2018) Sec. 3.1.3",
    })


def simulate_future_trend(fit, t_future, n_sims=200, seed=0):
    r"""Sec. 3.1.4: project the historical rate of change forward.

    Future changepoints arrive at the historical frequency
    :math:`S/T` with magnitudes from the inferred
    :math:`\mathrm{Laplace}(0,\lambda)`, so the interval widens with
    horizon. The paper does not claim exact coverage for this, and
    neither does the docstring.
    """
    tv = fit["t"]
    cps = fit["changepoints"]
    d = fit["deltas"]
    T = tv[-1] - tv[0]
    S = len(cps)
    if T <= 0.0:
        raise ValueError("prnFil: the history has no span")
    lam = (sum(abs(v) for v in d) / S) if S else 0.0
    rate = S / T
    rng = np.random.default_rng(seed)
    tf = [float(v) for v in t_future]
    sims = []
    for _ in range(int(n_sims)):
        nd = list(d)
        ncps = list(cps)
        for tv2 in tf:
            # one Bernoulli per future time step, at the historical
            # changepoint frequency
            if float(rng.uniform()) < rate * (tf[1] - tf[0]
                                              if len(tf) > 1 else 1.0):
                u = float(rng.uniform()) - 0.5
                mag = (-lam * math.copysign(1.0, u)
                       * math.log(1.0 - 2.0 * abs(u)) if lam > 0 else 0.0)
                ncps.append(tv2)
                nd.append(mag)
        sims.append(piecewise_trend(tf, fit["k"], fit["m"], nd, ncps))
    return sims


def trend_intervals(fit, t_future, level=0.8, n_sims=200, seed=0):
    """Quantile bands from the simulated future trends."""
    sims = simulate_future_trend(fit, t_future, n_sims=n_sims,
                                 seed=seed)
    H = len(t_future)
    lo_q = 0.5 - float(level) / 2.0
    hi_q = 0.5 + float(level) / 2.0
    lo, hi, med = [], [], []
    for h in range(H):
        col = sorted(s[h] for s in sims)
        lo.append(k.quantile7(col, lo_q))
        hi.append(k.quantile7(col, hi_q))
        med.append(k.quantile7(col, 0.5))
    return RichResult(payload={
        "estimate": med, "median": med, "lower": lo, "upper": hi,
        "width": [hi[h] - lo[h] for h in range(H)],
        "level": float(level), "n_sims": int(n_sims),
        "note": "the paper does not claim exact coverage for these; "
                "they indicate uncertainty and detect overfitting",
        "method": "trend forecast uncertainty by simulating future "
                  "changepoints, Taylor & Letham (2018) Sec. 3.1.4",
    })


def cheatsheet():
    return ("prnFil: lay down many candidate changepoints, let "
            "delta_j ~ Laplace(0, tau) decide. Small tau = straight "
            "trend, large tau = bends everywhere; training error falls "
            "and forecast intervals WIDEN as tau grows, which is the "
            "overfitting signal. Candidates only in the first 80 per "
            "cent: a changepoint near the end has no data after it and "
            "dominates every forecast.")


# compact alias per ledger/NAMING.md
selectchangepoints = select_changepoints

# public names resolved by fn/_lazy_map.json
prophet_changepoint = select_changepoints
