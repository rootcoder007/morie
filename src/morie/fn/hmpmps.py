# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apple MPS device placement plan."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_mps_acceleration"]

_UNSUPPORTED = ("float64", "complex64", "complex128", "int64")


def geron_mps_acceleration(tensor, dtype=None):
    """
    Apple MPS hardware acceleration (Metal Performance Shaders).

    Formula: move tensor to 'mps' device for GPU-equivalent ops on Apple Silicon

    No Metal call is made -- morie.fn is numpy-only -- and the array
    comes back on the CPU. What is real is the placement CHECK, which is
    where MPS actually bites: the Metal backend has no float64. A
    float64 tensor moved to 'mps' is silently demoted to float32, and
    everything downstream inherits about 7 significant digits instead of
    16. That demotion is performed here explicitly and its cost measured,
    so the precision loss is a number you can look at rather than a
    surprise in a loss curve.

    int64 has the same problem and is demoted to int32, which OVERFLOWS
    rather than rounding; that is reported separately because it is a
    correctness failure, not a precision one.

    Unified memory means there is no host-to-device copy on Apple
    silicon, so the transfer cost that dominates CUDA planning is absent
    -- the only thing to plan for is the dtype.

    Parameters
    ----------
    tensor : array-like
        Data to place.
    dtype : str, optional
        Force a target dtype instead of the automatic demotion.

    Returns
    -------
    result : RichResult
        Keys: tensor, source_dtype, dtype_on_device, downcast,
        max_abs_error, relative_error, overflow, unified_memory,
        estimate, n, method.

    Examples
    --------
    A float64 tensor is demoted to float32 with a measurable error:

    >>> r = geron_mps_acceleration(np.array([1.0 / 3.0, 2.0 / 3.0]))
    >>> r["source_dtype"], r["dtype_on_device"], bool(r["downcast"])
    ('float64', 'float32', True)
    >>> bool(0 < r["max_abs_error"] < 1e-7)
    True

    A float32 tensor is already native, so nothing changes:

    >>> f = geron_mps_acceleration(np.array([1.0, 2.0], dtype=np.float32))
    >>> bool(f["downcast"]), float(f["max_abs_error"])
    (False, 0.0)

    int64 demotion can overflow rather than round, and says so:

    >>> o = geron_mps_acceleration(np.array([2 ** 40], dtype=np.int64))
    >>> bool(o["overflow"])
    True

    References
    ----------
    Geron Ch 10
    """
    a = np.asarray(tensor)
    if a.size == 0:
        raise ValueError("geron_mps_acceleration: tensor is empty")
    src = np.dtype(a.dtype).name

    if dtype is not None:
        target = np.dtype(str(dtype).replace("torch.", ""))
    elif src in ("float64", "longdouble"):
        target = np.dtype(np.float32)
    elif src == "int64":
        target = np.dtype(np.int32)
    elif src.startswith("complex"):
        raise ValueError(f"geron_mps_acceleration: the Metal backend has no complex support, cannot place {src}")
    else:
        target = a.dtype
    if np.dtype(target).name in _UNSUPPORTED and dtype is not None:
        raise ValueError(f"geron_mps_acceleration: dtype {np.dtype(target).name} is not supported on the MPS device")

    out = a.astype(target)
    downcast = np.dtype(target) != a.dtype
    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(target, np.floating):
        err = float(np.max(np.abs(out.astype(np.float64) - a.astype(np.float64))))
        denom = float(np.max(np.abs(a.astype(np.float64))))
        rel = err / denom if denom > 0 else 0.0
        overflow = bool(np.any(~np.isfinite(out)))
    elif np.issubdtype(a.dtype, np.integer):
        back = out.astype(np.int64)
        overflow = bool(np.any(back != a.astype(np.int64)))
        err = float(np.max(np.abs(back - a.astype(np.int64)))) if overflow else 0.0
        rel = 0.0
    else:
        err, rel, overflow = 0.0, 0.0, False

    return RichResult(
        title="MPS placement plan",
        summary_lines=[("Source dtype", src), ("On device", np.dtype(target).name), ("Max abs error", err)],
        warnings=(["int64 does not fit int32: this demotion OVERFLOWS, it does not round"] if overflow else [])
        + (["no Metal call is made: morie.fn is numpy-only and the result stays on the CPU"]),
        interpretation="Metal has no float64; the demotion is the plan's real cost, and unified memory makes the copy free.",
        payload={
            "tensor": out,
            "source_dtype": src,
            "dtype_on_device": np.dtype(target).name,
            "downcast": bool(downcast),
            "max_abs_error": err,
            "relative_error": rel,
            "overflow": overflow,
            "unified_memory": True,
            "nbytes": int(out.nbytes),
            "executes_on_metal": False,
            "estimate": out,
            "n": int(a.size),
            "method": "MPS dtype placement plan with demotion error measured on the CPU",
        },
    )


def cheatsheet():
    return "hmpmps: Apple MPS placement plan and dtype demotion cost"
