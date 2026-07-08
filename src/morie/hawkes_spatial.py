# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatiotemporal (self-exciting) Hawkes point process.

Models the conditional intensity of clustered space-time events -- near-miss
airspace violations, crime, seismic aftershocks -- where each event transiently
raises the risk of further events nearby and soon after. The purely temporal
Hawkes lives in ``morie.hawkes``; this adds the spatial dimension:

    lambda(t,x,y) = mu + sum_{t_i<t} alpha*beta*exp(-beta(t-t_i))
                    * (1/(2*pi*sigma^2)) * exp(-((x-x_i)^2+(y-y_i)^2)/(2*sigma^2))

with background rate ``mu`` (per unit area-time), branching ratio ``alpha``
(alpha<1 for stability), temporal decay ``beta``, Gaussian spatial spread
``sigma``. The triggering kernel integrates to ``alpha`` in time and 1 in space
(Reinhart 2018). R parity: ``rmorie`` ``R/hawkes_spatial.R`` (``morie_hawkes_st_*``).

References
----------
Reinhart A (2018). A review of self-exciting spatio-temporal point processes.
*Statistical Science* 33(3), 299-318.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _check_params(p: dict[str, float]) -> None:
    if not {"mu", "alpha", "beta", "sigma"} <= set(p):
        raise ValueError("`params` must have mu, alpha, beta, sigma")
    if p["mu"] < 0 or p["alpha"] < 0 or p["beta"] <= 0 or p["sigma"] <= 0:
        raise ValueError("require mu>=0, alpha>=0, beta>0, sigma>0")


def _spatial(d2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-d2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def hawkes_st_intensity(events, t_q: float, x_q: float, y_q: float,
                        params: dict[str, float]) -> float:
    """Conditional intensity lambda(t_q, x_q, y_q). R parity: ``morie_hawkes_st_intensity``."""
    _check_params(params)
    t = np.asarray(events["t"], dtype=float)
    x = np.asarray(events["x"], dtype=float)
    y = np.asarray(events["y"], dtype=float)
    past = t < t_q
    lam = params["mu"]
    if past.any():
        dt = t_q - t[past]
        d2 = (x_q - x[past]) ** 2 + (y_q - y[past]) ** 2
        lam += float(np.sum(params["alpha"] * params["beta"] *
                            np.exp(-params["beta"] * dt) *
                            _spatial(d2, params["sigma"])))
    return float(lam)


def hawkes_st_loglik(events, params: dict[str, float],
                     end_time: float | None = None, area: float = 1.0) -> float:
    """Log-likelihood sum(log lambda) - compensator. R parity: ``morie_hawkes_st_loglik``.

    Compensator = mu*T*A + sum_i alpha(1 - exp(-beta(T - t_i))); the spatial
    kernel integrates to 1 over the plane (interior-region boundary effects
    neglected).
    """
    _check_params(params)
    t = np.asarray(events["t"], dtype=float)
    x = np.asarray(events["x"], dtype=float)
    y = np.asarray(events["y"], dtype=float)
    o = np.argsort(t)
    t, x, y = t[o], x[o], y[o]
    n = t.size
    T_h = float(np.max(t)) if end_time is None else float(end_time)
    if n == 0:
        return -params["mu"] * T_h * area

    loglam = 0.0
    for j in range(n):
        lam = params["mu"]
        if j > 0:
            dt = t[j] - t[:j]
            d2 = (x[j] - x[:j]) ** 2 + (y[j] - y[:j]) ** 2
            lam += np.sum(params["alpha"] * params["beta"] *
                          np.exp(-params["beta"] * dt) * _spatial(d2, params["sigma"]))
        loglam += np.log(lam)
    compensator = params["mu"] * T_h * area + \
        np.sum(params["alpha"] * (1 - np.exp(-params["beta"] * (T_h - t))))
    return float(loglam - compensator)


def hawkes_st_simulate(params: dict[str, float], end_time: float,
                       region, seed: int | None = None,
                       max_events: int = 100_000) -> pd.DataFrame:
    """Exact branching (immigrant-offspring) simulation. Requires alpha < 1.

    R parity: ``morie_hawkes_st_simulate``. ``region`` = (xmin, xmax, ymin, ymax).
    Returns a DataFrame with t, x, y, gen (0 = immigrant), sorted by t.
    """
    _check_params(params)
    if params["alpha"] >= 1:
        raise ValueError("simulation requires alpha < 1 (subcritical)")
    if len(region) != 4:
        raise ValueError("`region` must be (xmin, xmax, ymin, ymax)")
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = region
    area = (xmax - xmin) * (ymax - ymin)

    n_imm = rng.poisson(params["mu"] * area * end_time)
    t = rng.uniform(0, end_time, n_imm)
    x = rng.uniform(xmin, xmax, n_imm)
    y = rng.uniform(ymin, ymax, n_imm)
    gen = np.zeros(n_imm, dtype=int)
    all_t, all_x, all_y, all_g = [t], [x], [y], [gen]
    total = n_imm

    cur_t, cur_x, cur_y, cur_g = t, x, y, gen
    while cur_t.size > 0 and total < max_events:
        n_off = rng.poisson(params["alpha"], cur_t.size)
        keep = n_off > 0
        if not keep.any():
            break
        pt = np.repeat(cur_t[keep], n_off[keep])
        px = np.repeat(cur_x[keep], n_off[keep])
        py = np.repeat(cur_y[keep], n_off[keep])
        pg = np.repeat(cur_g[keep], n_off[keep])
        m = pt.size
        ot = pt + rng.exponential(1.0 / params["beta"], m)
        ox = px + rng.normal(0, params["sigma"], m)
        oy = py + rng.normal(0, params["sigma"], m)
        within = ot < end_time
        ot, ox, oy, og = ot[within], ox[within], oy[within], pg[within] + 1
        if ot.size == 0:
            break
        all_t.append(ot); all_x.append(ox); all_y.append(oy); all_g.append(og)
        total += ot.size
        cur_t, cur_x, cur_y, cur_g = ot, ox, oy, og

    t = np.concatenate(all_t); x = np.concatenate(all_x)
    y = np.concatenate(all_y); g = np.concatenate(all_g)
    o = np.argsort(t)
    return pd.DataFrame({"t": t[o], "x": x[o], "y": y[o], "gen": g[o]})


def hawkes_st_fit(events, end_time: float | None = None, area: float = 1.0,
                  start: dict[str, float] | None = None) -> dict[str, Any]:
    """Maximum-likelihood fit over (mu, alpha, beta, sigma) on the log scale.

    R parity: ``morie_hawkes_st_fit``. Check ``fit['params']['alpha'] < 1`` for
    a stable fit. Note: the background/branching split (mu vs alpha) is weakly
    identified at small samples and the likelihood neglects spatial edge
    effects, so interpret mu vs alpha cautiously and prefer large regions
    relative to sigma and long records.
    """
    t = np.asarray(events["t"], dtype=float)
    T_h = float(np.max(t)) if end_time is None else float(end_time)
    s = start or {"mu": max(t.size / (T_h * area), 1e-3),
                  "alpha": 0.3, "beta": 1.0, "sigma": 1.0}
    p0 = np.log([s["mu"], s["alpha"], s["beta"], s["sigma"]])

    def nll(lp):
        p = {"mu": np.exp(lp[0]), "alpha": np.exp(lp[1]),
             "beta": np.exp(lp[2]), "sigma": np.exp(lp[3])}
        try:
            val = hawkes_st_loglik(events, p, end_time=T_h, area=area)
        except (ValueError, FloatingPointError):
            return 1e10
        return 1e10 if not np.isfinite(val) else -val

    opt = minimize(nll, p0, method="L-BFGS-B")
    p = np.exp(opt.x)
    return {
        "params": {"mu": float(p[0]), "alpha": float(p[1]),
                   "beta": float(p[2]), "sigma": float(p[3])},
        "loglik": float(-opt.fun),
        "n": int(t.size),
        "convergence": 0 if opt.success else 1,
    }
