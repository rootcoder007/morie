# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pipeline parallelism: staged layers with microbatches."""

from . import _array_core as np

from ._richresult import RichResult
from .hmmpp import geron_model_parallelism

__all__ = ["geron_pipeline_parallelism"]


def geron_pipeline_parallelism(model, n_stages, n_microbatches=4):
    """
    Pipeline parallelism: partition layers across devices with microbatches.

    Formula: forward/backward staged across devices; microbatch pipeline

    The partition is DELEGATED to
    :func:`~morie.fn.hmmpp.geron_model_parallelism`; what pipelining adds
    is the fix for its central defect. Plain model parallelism leaves
    every stage but one idle; splitting the batch into M microbatches
    lets stage 2 start on microbatch 1 while stage 1 works on microbatch
    2, so all S stages run at once in steady state.

    The idle time does not vanish, it shrinks: filling and draining the
    pipe costs S-1 slots out of M+S-1, so the BUBBLE fraction is
    (S-1)/(M+S-1). More microbatches shrink it and cost activation
    memory -- the trade GPipe makes, and the number to look at before
    adding a stage.

    Parameters
    ----------
    model : sequence or mapping
        Per-layer parameter counts or weight arrays.
    n_stages : int
        Pipeline stages (>= 1).
    n_microbatches : int, default 4
        Microbatches per batch (>= 1).

    Returns
    -------
    result : RichResult
        Keys: assignment, stage_loads, bubble_fraction, utilisation,
        schedule, n_slots, estimate, n, method.

    Examples
    --------
    Four stages with four microbatches: 3 of 7 slots are bubble.

    >>> r = geron_pipeline_parallelism([1, 1, 1, 1], 4, n_microbatches=4)
    >>> round(float(r["bubble_fraction"]), 6), int(r["n_slots"])
    (0.428571, 7)

    Two stages, four microbatches: (2-1)/(4+2-1) = 0.2.

    >>> round(float(geron_pipeline_parallelism([1, 1, 1, 1], 2)["bubble_fraction"]), 6)
    0.2

    Sixteen microbatches over the same two stages nearly fill the pipe:

    >>> round(float(geron_pipeline_parallelism([1, 1, 1, 1], 2, 16)["utilisation"]), 6)
    0.941176

    The schedule says which microbatch each stage runs in each slot
    (-1 for a bubble):

    >>> geron_pipeline_parallelism([1, 1], 2, 2)["schedule"].tolist()
    [[0, 1, -1], [-1, 0, 1]]

    References
    ----------
    Geron Ch 17
    """
    base = geron_model_parallelism(model, n_stages)
    S = int(n_stages)
    M = int(n_microbatches)
    if M < 1:
        raise ValueError(f"geron_pipeline_parallelism: n_microbatches must be >= 1, got {n_microbatches!r}")

    slots = M + S - 1
    bubble = (S - 1) / slots
    sched = -np.ones((S, slots), dtype=int)
    for s in range(S):
        for m in range(M):
            sched[s, s + m] = m

    return RichResult(
        title="Pipeline parallelism",
        summary_lines=[("Stages", S), ("Microbatches", M), ("Bubble fraction", bubble)],
        interpretation="Bubble is (S-1)/(M+S-1): more microbatches buy utilisation with activation memory.",
        payload={
            "assignment": base["assignment"],
            "stage_loads": base["device_loads"],
            "max_load": base["max_load"],
            "imbalance": base["imbalance"],
            "bubble_fraction": bubble,
            "utilisation": 1.0 - bubble,
            "schedule": sched,
            "n_slots": int(slots),
            "n_microbatches": M,
            "estimate": bubble,
            "n": int(np.asarray(base["layer_sizes"]).size),
            "method": "Pipeline schedule over a contiguous stage partition (partition from morie.fn.hmmpp)",
        },
    )


def cheatsheet():
    return "hmppp: Pipeline parallelism, staged layers with microbatches"
