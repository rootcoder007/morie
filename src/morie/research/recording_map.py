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

__all__ = ["recording_map", "detection_rate_shift"]


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
