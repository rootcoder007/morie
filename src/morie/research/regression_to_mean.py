# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P19: regression to the mean at selected hot spots (``research/lean/P19Regression.lean``).

* ``Research.P19.exchange_mass`` / ``exchange_cross`` / ``indicator_bound``
* ``Research.P19.selected_change_nonpos`` / ``low_selected_change_nonneg``

R parity: ``rmorie`` ``R/regression_to_mean.R`` (``morie_regression_to_mean``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np

__all__ = ["regression_to_mean", "hotspot_shrinkage", "shrinkage_loss"]


def regression_to_mean(x1, x2, threshold, weights=None) -> dict:
    """Observed, mirror and symmetrised change on the places selected for a high first-period count.

    The symmetrised change is the same statistic on the data plus its period-swapped
    copy, an exchangeable population, so it is non-positive by
    ``Research.P19.selected_change_nonpos``.

    Examples
    --------
    >>> r = regression_to_mean([5, 1, 7, 2, 9, 3], [3, 2, 6, 2, 4, 5], threshold=4)
    >>> (r["n_selected"], round(r["selected_change"], 12), r["symmetrised_change"] <= 0)
    (3, -2.666666666667, True)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n = x1.shape[0]
    if x2.shape[0] != n:
        raise ValueError("x1 and x2 must have equal length")
    if np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
        raise ValueError("counts must not contain NA")
    try:
        c = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a single number") from exc
    if c != c:
        raise ValueError("threshold must be a single number")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0):
        raise ValueError("weights must be non-negative")
    sel1 = x1 > c
    sel2 = x2 > c
    low1 = x1 < c
    low2 = x2 < c
    if not np.any(sel1):
        raise ValueError("no place exceeds the threshold")
    d = x2 - x1
    sel_change = float(np.sum(w[sel1] * d[sel1]) / np.sum(w[sel1]))
    mirror = float(np.sum(w[sel2] * (-d)[sel2]) / np.sum(w[sel2])) if np.any(sel2) else float("nan")
    sym = float((np.sum(w[sel1] * d[sel1]) + np.sum(w[sel2] * (-d)[sel2])) / (np.sum(w[sel1]) + np.sum(w[sel2])))
    low_change = float(np.sum(w[low1] * d[low1]) / np.sum(w[low1])) if np.any(low1) else float("nan")
    if np.any(low1) or np.any(low2):
        low_sym = float(
            (np.sum(w[low1] * d[low1]) + np.sum(w[low2] * (-d)[low2])) / (np.sum(w[low1]) + np.sum(w[low2]))
        )
    else:
        low_sym = float("nan")
    return {
        "n_selected": int(np.sum(sel1)),
        "selected_change": sel_change,
        "mirror_change": mirror,
        "symmetrised_change": sym,
        "excess_over_symmetry": sel_change - sym,
        "low_selected_change": low_change,
        "low_symmetrised_change": low_sym,
        "theorems": [
            "Research.P19.exchange_mass",
            "Research.P19.exchange_cross",
            "Research.P19.indicator_bound",
            "Research.P19.selected_change_nonpos",
            "Research.P19.low_selected_change_nonneg",
        ],
    }


def _w_or_ones(weights, n):
    if weights is None:
        return [1.0] * n
    w = [float(x) for x in weights]
    if len(w) != n or any(math.isnan(x) or x < 0 for x in w) or math.fsum(w) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    return w


def hotspot_shrinkage(y, noise_variance, weights=None, B=None) -> dict:
    """Empirical-Bayes shrinkage of hot-spot counts: the expected size of the fall.

    ``Research.P19Shrinkage.loss_min`` / ``bstar_mem`` / ``predicted_fall``.

    Examples
    --------
    >>> s = hotspot_shrinkage([40, 12, 9, 25, 7, 31, 5, 18], noise_variance=18.375)
    >>> (round(s["B"], 12), round(s["predicted_fall"][0], 12), round(s["shrunk"][0], 12), s["total_variance"])
    (0.132686449284, 2.869344465757, 37.130655534243, 138.484375)
    """
    yy = [float(x) for x in y]
    if len(yy) < 2 or any(math.isnan(x) for x in yy):
        raise ValueError("y must be at least two numbers without NA")
    if noise_variance is None or math.isnan(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be a non-negative number")
    w = _w_or_ones(weights, len(yy))
    W = math.fsum(w)
    ybar = math.fsum(wi * yi for wi, yi in zip(w, yy)) / W
    total = math.fsum(wi * (yi - ybar) ** 2 for wi, yi in zip(w, yy)) / W
    signal = max(total - noise_variance, 0.0)
    Se = noise_variance * W
    St = signal * W
    Bstar = Se / (Se + St) if Se + St > 0 else 0.0
    if B is None:
        B = Bstar
    if B is None or math.isnan(B) or B < 0 or B > 1:
        raise ValueError("B must lie in [0, 1]")
    return {
        "mean": ybar,
        "total_variance": total,
        "noise_variance": float(noise_variance),
        "signal_variance": signal,
        "B": B,
        "B_star": Bstar,
        "shrunk": [(1 - B) * yi + B * ybar for yi in yy],
        "predicted_fall": [B * (yi - ybar) for yi in yy],
        "theorems": [
            "Research.P19Shrinkage.loss_min",
            "Research.P19Shrinkage.bstar_mem",
            "Research.P19Shrinkage.predicted_fall",
        ],
    }


def shrinkage_loss(theta, noise, weights=None, B=None) -> dict:
    """Loss of the shrinkage estimator against a known truth, next to the closed form of ``loss_eq``.

    ``Research.P19Shrinkage.loss_eq`` / ``loss_min`` / ``loss_bstar_eq`` / ``loss_bstar_le_raw``.

    Examples
    --------
    >>> l = shrinkage_loss([10, 20, 30, 40], [3, -3, -3, 3], B=0.5)
    >>> (l["loss"], l["closed_form"], round(l["B_star"], 12), round(l["loss_star"], 12), l["loss_raw"], l["noise_law"])
    (134.0, 134.0, 0.067164179104, 33.582089552239, 36.0, True)
    """
    th = [float(x) for x in theta]
    e = [float(x) for x in noise]
    if len(th) != len(e) or any(math.isnan(x) for x in th + e):
        raise ValueError("theta and noise must be numeric of equal length without NA")
    if B is None or isinstance(B, list | tuple) or math.isnan(B):
        raise ValueError("B must be a single number")
    w = _w_or_ones(weights, len(th))
    W = math.fsum(w)
    y = [a + b for a, b in zip(th, e)]
    ybar = math.fsum(wi * yi for wi, yi in zip(w, y)) / W
    tbar = math.fsum(wi * ti for wi, ti in zip(w, th)) / W
    Se = math.fsum(wi * ei**2 for wi, ei in zip(w, e))
    St = math.fsum(wi * (ti - tbar) ** 2 for wi, ti in zip(w, th))
    shrunk = [(1 - B) * yi + B * ybar for yi in y]
    law = abs(math.fsum(wi * ei for wi, ei in zip(w, e))) <= 1e-10 * max(1.0, W) and abs(
        math.fsum(wi * ti * ei for wi, ti, ei in zip(w, th, e))
    ) <= 1e-10 * max(1.0, math.fsum(abs(wi * ti) for wi, ti in zip(w, th)))
    Bstar = Se / (Se + St) if Se + St > 0 else 0.0
    return {
        "loss": math.fsum(wi * (si - ti) ** 2 for wi, si, ti in zip(w, shrunk, th)),
        "closed_form": (1 - B) ** 2 * Se + B**2 * St,
        "noise_law": law,
        "Se": Se,
        "Stheta": St,
        "B_star": Bstar,
        "loss_star": Se * St / (Se + St) if Se + St > 0 else 0.0,
        "loss_raw": Se,
        "theorems": [
            "Research.P19Shrinkage.loss_eq",
            "Research.P19Shrinkage.loss_min",
            "Research.P19Shrinkage.loss_bstar_eq",
            "Research.P19Shrinkage.loss_bstar_le_raw",
        ],
    }
