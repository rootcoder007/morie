# morie.fn -- function file (rootcoder007/morie)
"""Research P9: crime recording as a linear map.

Python twin of ``R/recording_map.R`` (``morie_recording_map``,
``morie_detection_rate_shift``); machine-checked in
``research/lean/P9Recording.lean``:

- ``Research.P9.total_invariant_of_colStochastic``: column-stochastic M keeps sum(M c) = sum(c)
- ``Research.P9.total_le_of_colSubstochastic``: column-substochastic M drops exactly the lost mass
- ``Research.P9.reclassification_moves_ratio``: re-classification moves ratios, not totals
- ``Research.P9.detection_rate_rises``: moving share q of a class (rate d) into a disposal
  detected with probability one raises the aggregate rate by q n (1 - d) / N
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["recording_map", "detection_rate_shift"]


def _rdiv(a, b):
    """R division: a / 0 is Inf (or NaN for 0 / 0) instead of raising."""
    if b:
        return a / b
    return math.copysign(math.inf, a) if a else math.nan


def recording_map(M, counts, labels=None):
    """Recorded counts from true counts through a recording matrix.

    ``M[i][j]`` is the share of true category-``j`` offences recorded as
    category ``i``. When every column of ``M`` sums to one (pure
    re-classification) the recorded total equals the true total, so
    downgrading shows only in category ratios; when columns sum to less
    than one (offences "cuffed" out of the notifiable set) the total falls
    by exactly the dropped mass.

    Parameters
    ----------
    M : list of list of float
        Square non-negative matrix (rows = recorded category) with column
        sums at most one.
    counts : sequence of float or dict
        True counts per category; a dict supplies the labels.
    labels : sequence of str, optional
        Category labels (default: the dict keys of ``counts``, if any).

    Returns
    -------
    RichResult
        ``recorded``, ``true_total``, ``recorded_total``, ``dropped``
        (``by_category`` and ``total``), ``regime`` (``"reclassification"``
        or ``"cuffing"``), ``ratio_true``, ``ratio_recorded`` (each category
        over the first), ``labels`` and ``theorems``.

    Examples
    --------
    >>> r = recording_map([[0.7, 0.0], [0.3, 1.0]], {"robbery": 100, "theft": 400})
    >>> r.recorded, r.recorded_total, r.regime
    ([70.0, 430.0], 500.0, 'reclassification')
    >>> r2 = recording_map([[0.7, 0.0], [0.2, 0.9]], [100, 400])
    >>> [round(v, 12) for v in r2.recorded], round(r2.dropped["total"], 12), r2.regime
    ([70.0, 380.0], 50.0, 'cuffing')
    """
    if isinstance(counts, dict):
        if labels is None:
            labels = list(counts.keys())
        counts = list(counts.values())
    counts = [float(v) for v in counts]
    M = [[float(v) for v in row] for row in M]
    m = len(M)
    if any(len(row) != m for row in M) or len(counts) != m:
        raise ValueError("M must be square with one column per category")
    cs = [math.fsum(M[i][j] for i in range(m)) for j in range(m)]
    if any(v < 0 for row in M for v in row) or any(c > 1 + 1e-12 for c in cs):
        raise ValueError("M must be non-negative with column sums at most one")
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    r = [math.fsum(M[i][j] * counts[j] for j in range(m)) for i in range(m)]
    dropped = [(1 - cs[j]) * counts[j] for j in range(m)]
    stochastic = all(abs(c - 1) < 1e-12 for c in cs)
    return RichResult(
        title="Recording map",
        payload={
            "recorded": r,
            "true_total": math.fsum(counts),
            "recorded_total": math.fsum(r),
            "dropped": {"by_category": dropped, "total": math.fsum(dropped)},
            "regime": "reclassification" if stochastic else "cuffing",
            "ratio_true": [_rdiv(c, counts[0]) for c in counts],
            "ratio_recorded": [_rdiv(v, r[0]) for v in r],
            "labels": list(labels) if labels is not None else None,
            "theorems": [
                "Research.P9.total_invariant_of_colStochastic",
                "Research.P9.total_le_of_colSubstochastic",
            ],
        },
    )


def detection_rate_shift(detected, total, n, d, q):
    """Detection-rate arithmetic of downgrade-and-caution.

    Moving a share ``q`` of an offence class of size ``n`` with detection
    rate ``d`` into a disposal detected with probability one raises the
    aggregate detection rate by exactly ``q n (1 - d) / N`` while the
    recorded total ``N`` is unchanged, so an aggregate clearance rate is no
    performance signal without the split by disposal type.

    Parameters
    ----------
    detected : float
        Aggregate detections before the move.
    total : float
        Recorded total ``N``.
    n : float
        Size of the class being moved.
    d : float
        Detection rate of that class before the move, in [0, 1).
    q : float
        Share of the class moved, in (0, 1].

    Returns
    -------
    RichResult
        ``rate_before``, ``rate_after``, ``rise`` and ``theorem``.

    Examples
    --------
    >>> s = detection_rate_shift(detected=2000, total=10000, n=1500, d=0.2, q=0.5)
    >>> s.rate_before, round(s.rate_after, 12), round(s.rise, 12)
    (0.2, 0.26, 0.06)
    """
    if total <= 0 or n <= 0 or n > total or detected < 0 or detected > total:
        raise ValueError("counts must satisfy 0 <= detected <= total and 0 < n <= total")
    if d < 0 or d >= 1 or q <= 0 or q > 1:
        raise ValueError("d must lie in [0, 1) and q in (0, 1]")
    rise = q * n * (1 - d) / total
    return RichResult(
        title="Detection rate after downgrade-and-caution",
        payload={
            "rate_before": detected / total,
            "rate_after": (detected + q * n * (1 - d)) / total,
            "rise": rise,
            "theorem": "Research.P9.detection_rate_rises",
        },
    )


def cheatsheet() -> str:
    return (
        "recording_map(M, counts) -> recorded = M c, totals, dropped mass, regime (Research P9)\n"
        "detection_rate_shift(detected, total, n, d, q) -> detection-rate rise q n (1-d) / N"
    )
