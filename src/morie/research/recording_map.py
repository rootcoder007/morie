# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P9: crime recording as a linear map (``research/lean/P9Recording.lean``).

* ``Research.P9.total_invariant_of_colStochastic``: column-stochastic M keeps the total
* ``Research.P9.total_le_of_colSubstochastic``: column-substochastic M drops exactly the lost mass
* ``Research.P9.reclassification_moves_ratio``
* ``Research.P9.detection_rate_rises``

R parity: ``rmorie`` ``R/recording_map.R`` (``morie_recording_map``, ``morie_detection_rate_shift``).
"""

from __future__ import annotations

from morie.fn import _array_core as np

__all__ = ["recording_map", "detection_rate_shift", "recording_regimes"]


def recording_map(M, counts, names=None) -> dict:
    """Recorded counts from true counts through a recording matrix.

    ``M[i, j]`` is the share of true category-j offences recorded as category i.

    Examples
    --------
    >>> r = recording_map([[0.7, 0], [0.3, 1]], [100, 400], names=["robbery", "theft"])
    >>> (r["recorded"], r["regime"], r["recorded_total"])
    ({'robbery': 70.0, 'theft': 430.0}, 'reclassification', 500.0)
    """
    M = np.asarray(M, dtype=float)
    c = np.asarray(counts, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or c.shape[0] != M.shape[1]:
        raise ValueError("M must be square with one column per category")
    colsum = M.sum(axis=0)
    if np.any(M < 0) or np.any(colsum > 1 + 1e-12):
        raise ValueError("M must be non-negative with column sums at most one")
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    r = M @ c
    if names is None:
        names = [str(i) for i in range(c.shape[0])]
    dropped = (1 - colsum) * c
    stochastic = bool(np.all(np.abs(colsum - 1) < 1e-12))
    return {
        "recorded": {names[i]: float(r[i]) for i in range(len(names))},
        "true_total": float(c.sum()),
        "recorded_total": float(r.sum()),
        "dropped": {
            "by_category": {names[i]: float(dropped[i]) for i in range(len(names))},
            "total": float(dropped.sum()),
        },
        "regime": "reclassification" if stochastic else "cuffing",
        "ratio_true": [float(v) for v in c / c[0]],
        "ratio_recorded": [float(v) for v in r / r[0]],
        "theorems": ["Research.P9.total_invariant_of_colStochastic", "Research.P9.total_le_of_colSubstochastic"],
    }


def detection_rate_shift(detected, total, n, d, q) -> dict:
    """Detection-rate arithmetic of downgrade-and-caution (``Research.P9.detection_rate_rises``).

    Examples
    --------
    >>> r = detection_rate_shift(detected=2000, total=10000, n=1500, d=0.2, q=0.5)
    >>> (r["rate_before"], r["rate_after"], r["rise"])
    (0.2, 0.26, 0.06)
    """
    detected = float(detected)
    total = float(total)
    n = float(n)
    d = float(d)
    q = float(q)
    if total <= 0 or n <= 0 or n > total or detected < 0 or detected > total:
        raise ValueError("counts must satisfy 0 <= detected <= total and 0 < n <= total")
    if d < 0 or d >= 1 or q <= 0 or q > 1:
        raise ValueError("d must lie in [0, 1) and q in (0, 1]")
    rise = q * n * (1 - d) / total
    return {
        "rate_before": detected / total,
        "rate_after": (detected + q * n * (1 - d)) / total,
        "rise": rise,
        "theorem": "Research.P9.detection_rate_rises",
    }


def recording_regimes(recorded, cuffed_share=0.1, true_total=None, names=None) -> dict:
    """Two recording regimes from one recorded series (``Research.P9.regimes_not_identified``,
    ``second_source_separates``).

    The identity map on true counts equal to the series and the map ``(1 - eps) I`` on
    ``recorded / (1 - eps)`` both reproduce the series; only a second source that fixes the
    true total tells them apart.

    Examples
    --------
    >>> two = recording_regimes([70, 430], cuffed_share=0.2)
    >>> (two["reclassification"]["true_total"], two["cuffing"]["true_total"], two["identified_from_series"])
    (500.0, 625.0, False)
    >>> recording_regimes([70, 430], true_total=550)["second_source"]["regime"]
    'cuffing'
    """
    r = np.asarray(recorded, dtype=float)
    if r.ndim != 1 or r.shape[0] == 0 or np.any(np.isnan(r)) or np.any(r < 0):
        raise ValueError("recorded must be non-negative counts")
    if not (0 < cuffed_share < 1):
        raise ValueError("cuffed_share must lie in (0, 1)")
    k = r.shape[0]
    if names is None:
        names = [str(i) for i in range(k)]
    ident = np.eye(k)
    cuff = (1 - cuffed_share) * ident
    counts_cuff = r / (1 - cuffed_share)
    recorded_total = float(r.sum())
    second = None
    if true_total is not None:
        tt = float(true_total)
        if np.isnan(tt) or tt < 0:
            raise ValueError("true_total must be a single non-negative number")
        tol = 1e-9 * max(1.0, tt)
        if abs(recorded_total - tt) <= tol:
            regime = "reclassification"
        elif recorded_total < tt:
            regime = "cuffing"
        else:
            regime = "inconsistent"
        second = {
            "true_total": tt,
            "dropped_mass": max(0.0, tt - recorded_total),
            "regime": regime,
            "consistent": recorded_total <= tt + tol,
        }
    return {
        "recorded": {names[i]: float(r[i]) for i in range(k)},
        "recorded_total": recorded_total,
        "reclassification": {
            "M": [[float(v) for v in row] for row in ident],
            "counts": {names[i]: float(r[i]) for i in range(k)},
            "true_total": recorded_total,
        },
        "cuffing": {
            "M": [[float(v) for v in row] for row in cuff],
            "counts": {names[i]: float(counts_cuff[i]) for i in range(k)},
            "true_total": float(counts_cuff.sum()),
            "share": float(cuffed_share),
        },
        "identified_from_series": False,
        "second_source": second,
        "theorems": ["Research.P9.regimes_not_identified", "Research.P9.second_source_separates"],
    }
