# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P13: pooling evaluations across sites (``research/lean/P13Meta.lean``; DerSimonian & Laird 1986).

* ``Research.P13.truncation_bias`` / ``pos_part_ge`` / ``pos_part_pos`` / ``dl_biased_under_homogeneity``
* ``Research.P13.tauDL_nonneg`` / ``tauDL_eq_zero_iff``
* ``Research.P13.re_var_ge`` / ``re_var_eq_iff``

R parity: ``rmorie`` ``R/meta_pooling.R`` (``morie_meta_random_effects``, ``morie_meta_dl_bias``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn._rng import normal_quantile, random_uniform
from morie.fn._stats_core import norm

__all__ = ["meta_random_effects", "meta_dl_bias"]


def meta_random_effects(estimates, variances, level=0.95) -> dict:
    """Fixed-effect and DerSimonian-Laird random-effects pooling, with what the truncation does.

    Examples
    --------
    >>> m = meta_random_effects([-0.25, -0.10, -0.40, 0.05, -0.30], [0.010, 0.020, 0.015, 0.030, 0.012])
    >>> (round(m["tau2"], 12), m["variance_ratio"] >= 1, m["truncated"])
    (0.006967741935, True, False)
    """
    est = np.asarray(estimates, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = est.shape[0]
    if v.shape[0] != k:
        raise ValueError("estimates and variances must have equal length")
    if k < 2:
        raise ValueError("need at least two sites")
    if np.any(np.isnan(est)) or np.any(np.isnan(v)) or np.any(v <= 0):
        raise ValueError("variances must be positive and nothing missing")
    if level <= 0 or level >= 1:
        raise ValueError("level must lie in (0, 1)")
    w = 1 / v
    theta_fe = float(np.sum(w * est) / np.sum(w))
    q = float(np.sum(w * (est - theta_fe) ** 2))
    c_dl = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c_dl)
    w_re = 1 / (v + tau2)
    theta_re = float(np.sum(w_re * est) / np.sum(w_re))
    var_fe = 1 / float(np.sum(w))
    var_re = 1 / float(np.sum(w_re))
    z = float(norm.ppf(1 - (1 - level) / 2))

    def ci(e, vv):
        return {"lower": e - z * math.sqrt(vv), "upper": e + z * math.sqrt(vv)}

    return {
        "k": k,
        "fixed": {"estimate": theta_fe, "variance": var_fe, "se": math.sqrt(var_fe), "ci": ci(theta_fe, var_fe)},
        "random": {"estimate": theta_re, "variance": var_re, "se": math.sqrt(var_re), "ci": ci(theta_re, var_re)},
        "Q": q,
        "df": k - 1,
        "c": c_dl,
        "tau2": tau2,
        "truncated": q <= k - 1,
        "variance_ratio": var_re / var_fe,
        "weights": {"fixed": [float(x) for x in w / np.sum(w)], "random": [float(x) for x in w_re / np.sum(w_re)]},
        "theorems": [
            "Research.P13.tauDL_nonneg",
            "Research.P13.tauDL_eq_zero_iff",
            "Research.P13.re_var_ge",
            "Research.P13.re_var_eq_iff",
        ],
    }


def meta_dl_bias(variances, n_draws=2000, seed=0) -> dict:
    """Size of the DerSimonian-Laird truncation bias under homogeneity (shared Philox stream).

    Examples
    --------
    >>> b = meta_dl_bias([0.010, 0.020, 0.015, 0.030, 0.012], n_draws=500, seed=1)
    >>> (b["mean_tau2"] > 0, 0 < b["share_positive"] < 1)
    (True, True)
    """
    v = np.asarray(variances, dtype=float)
    k = v.shape[0]
    if k < 2 or np.any(np.isnan(v)) or np.any(v <= 0):
        raise ValueError("variances must be at least two positive numbers")
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError("n_draws must be a positive integer")
    seed = float(seed)
    if seed != seed or seed < 0:
        raise ValueError("seed must be a single non-negative number")
    u = random_uniform(k * n_draws, seed=int(seed), stream=0)
    z = np.array([float(normal_quantile(float(x))) for x in u])
    w = 1 / v
    c_dl = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    sd = np.sqrt(v)
    tau2 = []
    q_all = []
    ratio = []
    fe_var = 1 / float(np.sum(w))
    for d in range(n_draws):
        est = sd * z[d * k : (d + 1) * k]
        theta_fe = float(np.sum(w * est) / np.sum(w))
        q = float(np.sum(w * (est - theta_fe) ** 2))
        t2 = max(0.0, (q - (k - 1)) / c_dl)
        tau2.append(t2)
        q_all.append(q)
        ratio.append((1 / float(np.sum(1 / (v + t2)))) / fe_var)
    return {
        "k": k,
        "mean_tau2": sum(tau2) / n_draws,
        "share_positive": sum(1 for t in tau2 if t > 0) / n_draws,
        "mean_variance_ratio": sum(ratio) / n_draws,
        "mean_Q": sum(q_all) / n_draws,
        "n_draws": n_draws,
        "theorems": [
            "Research.P13.truncation_bias",
            "Research.P13.pos_part_pos",
            "Research.P13.dl_biased_under_homogeneity",
        ],
    }
