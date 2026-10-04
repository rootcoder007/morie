# morie.fn -- function file (rootcoder007/morie)
"""Heart rate variability metrics."""

from __future__ import annotations

from . import _array_core as np
from ._containers import DescriptiveResult


def hrv_metrics_fn(
    rr_intervals: np.ndarray,
) -> DescriptiveResult:
    """Heart-rate variability metrics from a series of RR intervals."""
    from morie._bioplot import hrv_metrics

    rr_intervals = np.asarray(rr_intervals, dtype=float)
    metrics = hrv_metrics(rr_intervals)
    return DescriptiveResult(
        name="hrv_metrics",
        value=metrics["sdnn"],
        extra=metrics,
    )


hrvmt = hrv_metrics_fn


def cheatsheet() -> str:
    return "hrv_metrics_fn({}) -> Heart rate variability metrics."


# compact alias per ledger/NAMING.md
hrvmetricsfn = hrv_metrics_fn
