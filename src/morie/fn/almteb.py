# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MTEB-style aggregation: mean over categories of mean over tasks
(Muennighoff et al. 2023; Alammar Ch 8)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_mteb_benchmark_score"]


def alammar_mteb_benchmark_score(task_scores, category_map):
    """Category means first, then the mean of means -- NOT the flat
    task mean. The two differ whenever categories have unequal sizes,
    and both are returned so the gap is visible.

    Examples
    --------
    >>> out = alammar_mteb_benchmark_score(
    ...     {"a": 1.0, "b": 0.0, "c": 0.5},
    ...     {"a": "x", "b": "x", "c": "y"})
    >>> out["estimate"]
    0.5
    """
    if not task_scores:
        raise ValueError("no task scores supplied.")
    missing = [t for t in task_scores if t not in category_map]
    if missing:
        raise ValueError(f"tasks {missing} have no category.")
    by_cat = {}
    for t, s in task_scores.items():
        by_cat.setdefault(category_map[t], []).append(float(s))
    cat_means = {c: float(np.mean(v)) for c, v in sorted(by_cat.items())}
    overall = float(np.mean(list(cat_means.values())))
    flat = float(np.mean([float(v) for v in task_scores.values()]))
    return RichResult(payload={
        "estimate": overall, "category_means": cat_means,
        "flat_task_mean": flat,
        "weighting_matters": abs(overall - flat) > 1e-12,
        "n": len(task_scores),
        "method": "MTEB mean-of-category-means (Muennighoff et al. 2023)"})


def cheatsheet():
    return "almteb: mean over categories of task means, flat mean shown too"
