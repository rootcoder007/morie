# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P19: regression to the mean at selected hot spots (``research/lean/P19Regression.lean``).

* ``Research.P19.exchange_mass`` / ``exchange_cross`` / ``indicator_bound``
* ``Research.P19.selected_change_nonpos`` / ``low_selected_change_nonneg``

R parity: ``rmorie`` ``R/regression_to_mean.R`` (``morie_regression_to_mean``).
"""

from __future__ import annotations

from morie.fn import _array_core as np

__all__ = ["regression_to_mean"]


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
